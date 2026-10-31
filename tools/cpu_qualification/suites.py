"""Native numerical and packaged CLI checks against frozen independent data."""
import math

from .common import ROOT, compare, pinned, read, require, sha, vector, write
from .profiles import PROFILES


def generated(actual, case, image=False):
    expected_tokens = case.get('greedy_token_ids', case.get('generated_token_ids'))
    expected_text = case.get('greedy_text', case.get('text'))
    require(actual['token_ids'] == expected_tokens and actual['text'] == expected_text,
            'exact greedy continuation mismatch: ' + case['id'])


def cli_generation(context, name, model, case, images=None, gguf=False):
    arguments = ['chat' if images else 'run', str(model), '--standalone', '--context',
                 '512' if gguf else '256', '--max-tokens',
                 str(len(case.get('greedy_token_ids', case.get('generated_token_ids')))),
                 '--profile', 'reference', '--threads', '1']
    if images:
        arguments += ['--prompt', case['text']]
        for path in images:
            arguments += ['--image', str(path)]
    elif 'messages' in case:
        path = context.directory / (name + '-messages.json')
        write(path, case['messages'])
        arguments += ['--messages', str(path)]
    else:
        arguments += ['--prompt', case['prompt']]
    result = context.cli(name, arguments)
    generated(result, case)
    return result


def gguf(context, model):
    golden = read(ROOT / 'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json')
    prompts = {c['id']: c for c in read(ROOT / 'fixtures/prompts.json')}
    for case in golden['cases']:
        messages = prompts[case['id']].get('messages')
        arguments = ['tokenize', str(model)]
        if messages:
            path = context.directory / (case['id'] + '-tokenize.json')
            write(path, messages)
            arguments += ['--messages', str(path)]
        else:
            arguments += ['--prompt', prompts[case['id']]['prompt']]
        tokenized = context.cli('tokens-' + case['id'], arguments)
        require(tokenized['prompt'] == case['prompt'] and tokenized['token_ids'] == case['prompt_token_ids'],
                'GGUF token/template mismatch: ' + case['id'])
        cli_generation(context, 'gguf-' + case['id'], model,
                       {**case, **({'messages': messages} if messages else {})}, gguf=True)
    actual = context.command('gguf-trace', [str(context.binary), 'trace', str(model), '--prompt', 'Hello', '--context', '512'])
    expected = [read_line for read_line in context.json_lines(context.references / 'references/gguf/reference-trace.jsonl')]
    require(len(actual) == len(expected) == 25, 'GGUF trace frame count mismatch')
    tolerance = read(ROOT / 'fixtures/models.json')['oracle']['tolerances']
    comparisons = []
    for a, b in zip(actual, expected):
        require(a['position'] == b['position'] and a['name'] == b['name'], 'GGUF checkpoint identity mismatch')
        comparisons.append({'name': a['name'], **compare(a['values'], b['values'], tolerance['logits_absolute'],
                                                       tolerance['logits_relative'], tolerance['normalized_l2'])})
    return {'cases': len(golden['cases']), 'full_arrays': comparisons}


def embedding(context, suite, model):
    entry = next(e for e in read(ROOT / 'fixtures/embedding/models.json')['models'] if e['file'] == model.name)
    ref = read(ROOT / entry['evidence'])
    tolerance = read(ROOT / 'fixtures/embedding/reference-contract.json')['tolerances']
    inputs = [c['input'] for c in ref['cases']]
    request = {'inputs': inputs, 'options': {'profile': 'reference', 'threads': 1}, 'tokenize_only': True}
    tokens = context.probe('embedding_probe', suite + '-tokens', model, request)
    require(tokens[0]['tokens'] == [[c['token_ids'], False] for c in ref['cases']], 'embedding tokenizer mismatch')

    def result(events):
        output = next(e['result'] for e in events if e['event'] == 'result')
        cleanup = next(e['memory'] for e in events if e['event'] == 'cleanup')
        require(cleanup['session_buffer_capacity_bytes'] == 0 and cleanup['peak_reserved'] <= cleanup['limit']
                and cleanup['reserved'] + output['plan']['output_bytes'] == output['memory']['reserved'], 'embedding cleanup mismatch')
        require(output['dimension'] == 768, 'embedding dimension mismatch')
        return output

    def numeric(a, b, batch=False):
        return compare(a, b, tolerance['batch_absolute' if batch else 'absolute'],
                       tolerance['batch_relative' if batch else 'relative'], None if batch else tolerance['normalized_l2'])

    comparisons, singles = [], []
    for case in ref['cases']:
        events = context.probe('embedding_probe', suite + '-' + case['id'], model,
                               {'inputs': [case['input']], 'options': request['options'], 'trace': True})
        output = result(events)
        require(len(output['data']) == 1 and output['data'][0]['prompt_tokens'] == len(case['token_ids'])
                and not output['data'][0]['truncated'], 'embedding input/usage mismatch')
        values = output['data'][0]['embedding']
        singles.append(values)
        comparisons.append(numeric(values, case['normalized']))
        require(abs(math.sqrt(sum(x * x for x in values)) - 1) <= tolerance['unit_norm_absolute'], 'embedding normalization mismatch')
        numeric(next(e['values'] for e in events if e.get('name') == 'pooled'), case['raw_pooled'])
        for checkpoint in case['checkpoints']:
            frame = next(e for e in events if e.get('name') == checkpoint['name'])
            require(frame['rows'] == checkpoint['rows'] and frame['width'] == checkpoint['width'], 'embedding checkpoint shape mismatch')
            numeric(frame['values'][:768], checkpoint['first_row'])
            numeric(frame['values'][-768:], checkpoint['last_row'])
    configurations = [('reference', 1, 4), ('balanced', 2, 4), ('performance', 4, 4), ('low-memory', 1, 4), ('balanced', 2, 1)]
    for profile, threads, batch in configurations:
        output = result(context.probe('embedding_probe', f'{suite}-{profile}-{threads}-{batch}', model,
                                     {'inputs': [inputs[i] for i in ref['batch_indices']],
                                      'options': {'profile': profile, 'threads': threads, 'batch_size': batch}}))
        require(len(output['data']) == len(ref['batch_items']) and output['usage']['prompt_tokens'] ==
                sum(len(c['token_ids']) for c in ref['batch_items']), 'batch count/usage mismatch')
        for index, (actual, expected) in enumerate(zip(output['data'], ref['batch_items'])):
            require(actual['index'] == index and actual['prompt_tokens'] == len(expected['token_ids'])
                    and not actual['truncated'], 'batch ordering/truncation mismatch')
            numeric(actual['embedding'], expected['normalized'])
            numeric(actual['embedding'], singles[ref['batch_indices'][index]], batch=True)
    cli = context.cli(suite + '-packaged', ['embed', str(model), '--standalone', '--input', inputs[0], '--profile', 'reference', '--threads', '1'])
    numeric(cli['data'][0]['embedding'], ref['cases'][0]['normalized'])
    return {'cases': len(inputs), 'batch_configurations': configurations, 'vectors': comparisons}


def qwen35(context, suite, model):
    profile = PROFILES[suite]
    image = profile['image']
    ref = read(ROOT / profile['reference'])
    contract = read(ROOT / 'fixtures/qwen35/vision/contract-v1.json')
    images = {i['id']: pinned(ROOT / i['file'], {**i, 'bytes': (ROOT / i['file']).stat().st_size}) for i in contract['images']}
    results = []
    for case in ref['cases']:
        for threads in ((1,) if image else (1, 2)):
            name = suite + '-' + case['id'] + '-' + str(threads)
            directory = context.directory / name
            directory.mkdir()
            selected = model
            if threads == 2 and profile['layout'] == 'publisher':
                selected = model / 'model.safetensors-00001-of-00001.safetensors'
            elif threads == 2 and profile['layout'] == 'sharded':
                selected = model / 'model.safetensors.index.json'
            request = ({'images': [str(images[i]) for i in case['images']], 'text': case['text'], 'teacher': case['greedy_token_ids'][:-1]} if image else
                       {'messages': case['messages'], 'teacher_tokens': case['greedy_token_ids'][:-1], 'threads': threads,
                        'max_tokens': len(case['greedy_token_ids']), 'trace_directory': str(directory), 'lifecycle': threads == 1})
            actual = context.probe('qwen35_image_probe' if image else 'qwen35_probe', name, selected, request,
                                   [str(directory)] if image else [])[0]
            require(actual['prompt'] == case['prompt'] and actual['prompt_token_ids'] == case['prompt_token_ids'], 'Qwen3.5 tokenizer/template mismatch')
            generated(actual['generation'], case)
            require(''.join(e['text'] for e in actual['events']) == case['greedy_text'], 'incremental text mismatch')
            resident = 256 << 20 if image else actual['resident']['reserved']
            require(actual['after']['reserved'] == resident and actual['after']['session_buffer_capacity_bytes'] == 0, 'Qwen3.5 cleanup mismatch')
            comparisons = []

            def numeric(path, expected, kind):
                reference = context.references / 'references' / profile['arrays'] / (expected['name'] + '.f32')
                a, b = vector(path), vector(reference)
                length = math.prod(expected['shape']) if image else expected['length']
                require(len(b) == length, 'reference shape mismatch')
                limits = contract['tolerances'] if image else ref['tolerances']
                prefix = kind if image else 'logits'
                comparisons.append({'name': expected['name'], 'actual_sha256': sha(path), **compare(a, b, limits[prefix + '_absolute'], limits[prefix + '_relative'],
                                                                       limits[kind + '_normalized_l2'] if image else limits['normalized_l2'])})

            if image:
                for index, key in enumerate(case['images']):
                    for expected in ref['encoder_checkpoints']:
                        if expected['name'].startswith(key + '-') and not expected['name'].endswith('-patches'):
                            numeric(directory / (str(index) + '-' + expected['name'][len(key) + 1:] + '.f32'), expected, 'vision')
            last = len(case['prompt_token_ids']) - 1
            for index, expected in enumerate(case['checkpoints']):
                numeric(directory / f'{last}-layer.{index}.f32', expected, 'language')
            for index, expected in enumerate(case['teacher_forced']):
                numeric(directory / f'{last + index}-logits.f32', expected, 'language')
            layout = profile['layout'] if profile['layout'] != 'publisher' else ('indexed' if threads == 1 else 'single-file')
            results.append({'case': case['id'], 'threads': threads, 'layout': layout, 'arrays': comparisons})
        cli_generation(context, suite + '-cli-' + case['id'], model, case, [images[i] for i in case['images']] if image else None)
    return {'runs': results, 'publisher_framework_parity': False}
