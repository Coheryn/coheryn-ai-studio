#!/usr/bin/env python3
"""Validate frozen image-conditioned reference data offline, without any executor."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_qwen_language(original):
    original_path=ROOT/'fixtures/vlm/image-conditioned-reference-v1.json'
    corrected_path=ROOT/'fixtures/vlm/qwen2-image-conditioned-reference-v2.json'
    corrected=json.loads(corrected_path.read_text())
    layers=json.loads((ROOT/'fixtures/vlm/qwen2-language-reference-v2.json').read_text())
    diagnostic=json.loads((ROOT/'fixtures/vlm/qwen2-language-reference-v1.json').read_text())
    independent=json.loads((ROOT/'fixtures/vlm/qwen2-language-independent-v2.json').read_text())
    source=ROOT/'resources/tool-assessment/qwen-language-source-record.json'
    record=json.loads(source.read_text())
    assert record['preparation_source_sha256']==sha(ROOT/'resources/tool-assessment/prepare_qwen_language_assessment.py')
    assert record['source_before_sha256']!=record['source_after_sha256']
    alignment=json.loads((ROOT/'fixtures/vlm/qwen2-language-alignment-v2.json').read_text())
    for field in ['original_reference','corrected_reference','source_configuration','correction_source_record','independent_check']:
        assert alignment[field+'_sha256']==sha(ROOT/alignment[field])
    assert record['source_before_sha256']==alignment['original_source_file_sha256']
    assert alignment['original_logged_epsilon']==0 and alignment['corrected_logged_epsilon']==1e-6
    assert json.loads((ROOT/alignment['source_configuration']).read_text())['rms_norm_eps']==alignment['published_rms_norm_epsilon']==1e-6

    for manifest in [corrected,layers]:
        assert manifest['schema_version']==2
        assert manifest['reference_revision']==original['reference_revision']
        assert manifest['original_reference_sha256']==sha(original_path)
        assert manifest['source_record_sha256']==sha(source)
        assert manifest['capture_source_sha256']==sha(ROOT/'resources/tool-assessment/capture_qwen_corrected_language.py')
        assert manifest['wrapper_source_sha256']==sha(ROOT/'resources/tool-assessment/vlm_language_oracle.cpp')
        assert manifest['base_wrapper_sha256']==sha(ROOT/'resources/tool-assessment/vlm_oracle.cpp')
        assert manifest['native_image_understanding_qualified'] is False
        assert len(manifest['cases'])==6
    old={c['id']:c for c in original['cases'] if c['model']=='qwen2-vl-2b'}
    assert {c['id'] for c in corrected['cases']}==old.keys()
    for case in corrected['cases']:
        previous=old[case['id']]
        for field in ['messages','identity','rendered_prompt','marker_prompt','template_sha256','encoded','chunks','staging_sha256']:
            assert case[field]==previous[field],f'correction changed independent input: {field}'
        tensors={t['name']:t for t in case['tensors']}
        for tensor in previous['tensors']:
            if tensor['name'].startswith('image_'):
                assert tensors[tensor['name']]==tensor,'epsilon correction changed vision features'
        assert case['language_reference']['source_record_sha256']==sha(source)
        assert case['language_reference']['library_sha256']==record['library_sha256']
        assert case['language_reference']['rms_norm_epsilon']==1e-6
        generation=case['generation']
        assert 1<=len(generation)<=8
        assert case['decoded']['generated_token_ids']==[g['token_id'] for g in generation]
        for i,g in enumerate(generation):
            assert g['greedy_step']==i and (not g['eos'] or i+1==len(generation))
            assert f'logits_{i}' in tensors
        assert case['output_bytes_hex']==''.join(g['piece_hex'] for g in generation)
        assert case['text']==bytes.fromhex(case['output_bytes_hex']).decode('utf8',errors='replace')
        checkpoint=next(c for c in layers['cases'] if c['id']==case['id'])
        assert len(checkpoint['tensors'])==28
        assert {t['native_checkpoint'] for t in checkpoint['tensors']}=={f'prompt_blk.{i}.output' for i in range(28)}
        for tensor in [*case['tensors'],*checkpoint['tensors']]:
            assert math.prod(tensor['shape'])==tensor['count']>0
            assert len(tensor['sha256'])==64
            assert all(math.isfinite(tensor[field]) for field in ['sum','l2','minimum','maximum'])
            assert all(0<=sample['index']<tensor['count'] and math.isfinite(sample['value']) for sample in tensor['samples'])
    assert diagnostic['capture_source_sha256']==sha(ROOT/'resources/tool-assessment/capture_vlm_language.py')
    assert diagnostic['wrapper_source_sha256']==sha(ROOT/'resources/tool-assessment/vlm_language_oracle.cpp')
    assert len(diagnostic['cases'])==6
    assert independent['source_sha256']==sha(ROOT/'resources/tool-assessment/qwen_layer_numpy.py')
    assert independent['math_source_sha256']==sha(ROOT/'resources/tool-assessment/qwen_scalar_math.c')
    assert independent['base_math_source_sha256']==sha(ROOT/'resources/tool-assessment/scalar_math.c')
    assert independent['tolerances']=={'absolute':0.003,'relative':0.001,'normalized_l2':0.0005}
    assert len(independent['comparisons'])==2
    for comparison,manifest in zip(independent['comparisons'],[diagnostic,layers]):
        red=next(c for c in manifest['cases'] if c['id']=='left-red')
        tensor=next(t for t in red['tensors'] if t.get('native_checkpoint')=='prompt_blk.0.output' or t['name'].endswith('_l_out-0'))
        assert comparison['reference_sha256']==tensor['sha256']==comparison['independent_sha256']
        assert comparison['outside_tolerance']==comparison['normalized_l2']==0
    assert independent['comparisons'][0]['epsilon']==0
    assert abs(independent['comparisons'][1]['epsilon']-1e-6)<1e-12


def main():
    path=ROOT/'fixtures/vlm/image-conditioned-reference-v1.json'
    reference=json.loads(path.read_text())
    requests_path=ROOT/'fixtures/vlm/requests-v1.json'
    requests=json.loads(requests_path.read_text())
    models=json.loads((ROOT/'fixtures/vlm/models.json').read_text())
    selected={m['id']:m for m in models['models']}
    expected={(group['model'],case['id']):case for group in requests['models'] for case in group['cases']}
    cases={(c['model'],c['id']):c for c in reference['cases']}
    assert len(cases)==len(reference['cases'])==len(expected)==8 and cases.keys()==expected.keys()
    assert reference['requests_sha256']==sha(requests_path)
    assert reference['capture_source_sha256']==sha(ROOT/'resources/tool-assessment/capture_vlm.py')
    assert reference['reference_source_record']=='resources/tool-assessment/vision-corrected-source-record.json'
    assert reference['reference_source_sha256']==sha(ROOT/reference['reference_source_record'])
    assert reference['reference_revision']==models['oracle']['revision']
    assert reference['native_image_understanding_qualified'] is False
    assert reference['settings']==requests['settings']
    original_vision=json.loads((ROOT/'fixtures/vlm/vision-reference.json').read_text())
    features={}
    position_fixture=json.loads((ROOT/'fixtures/vlm/positions-v1.json').read_text())
    assert position_fixture['wrapper_source_sha256']==sha(ROOT/'resources/tool-assessment/vlm_oracle.cpp')
    assert position_fixture['freeze_source_sha256']==sha(ROOT/'resources/tool-assessment/freeze_vlm_positions.py')
    total_spans=0
    for key,case in cases.items():
        model=selected[key[0]]
        components={c['role']:c for c in model['components']}
        identity=case['identity']
        assert identity['language_sha256']==components['language']['sha256']
        assert identity['vision_sha256']==components['vision-projector']['sha256']
        assert identity['wrapper_source_sha256']==sha(ROOT/'resources/tool-assessment/vlm_oracle.cpp')
        assert identity['settings']==requests['settings']
        assert case['messages']==expected[key]['messages']
        assert case['template_sha256']==sha(ROOT/'fixtures/vlm'/key[0]/'chat_template.json')
        assert identity['marker_prompt_sha256']==hashlib.sha256(case['marker_prompt'].encode()).hexdigest()
        expected_marker=case['rendered_prompt'].replace('<image>','<__media__>') if key[0]=='llava-1.5-7b' else case['rendered_prompt'].replace('<|vision_start|><|image_pad|><|vision_end|>','<__media__>')
        assert case['marker_prompt']==expected_marker
        images=identity['images']
        assert case['marker_prompt'].count('<__media__>')==len(images)
        expected_images=[p['fixture'] for message in expected[key]['messages'] if isinstance(message['content'],list) for p in message['content'] if p['type']=='image']
        assert len(expected_images)==len(images)
        for image,fixture in zip(images,expected_images):
            assert image['canonical']=='fixtures/vlm/preprocessing/canonical/'+fixture
            assert sha(ROOT/image['canonical'])==image['sha256']
        chunks=[c for c in case['chunks'] if 'chunk' in c]
        physical=position=0
        image_index=0
        tensors={t['name']:t for t in case['tensors']}
        assert len(tensors)==len(case['tensors'])
        for tensor in tensors.values():
            assert math.prod(tensor['shape'])==tensor['count'] and tensor['count']>0
            assert len(tensor['sha256'])==64
            assert all(math.isfinite(tensor[field]) for field in ['sum','l2','minimum','maximum'])
            assert all(0<=s['index']<tensor['count'] and math.isfinite(s['value']) for s in tensor['samples'])
        for index,chunk in enumerate(chunks):
            assert chunk['chunk']==index and chunk['start_position']==position
            assert chunk['tokens']>0 and chunk['position_count']>0
            if chunk['type']==0:assert len(chunk['token_ids'])==chunk['tokens']==chunk['position_count']
            else:
                assert chunk['type']==1 and chunk['image_index']==image_index
                assert tensors[f'image_{image_index}_embeddings']['shape']==[1536 if key[0]=='qwen2-vl-2b' else 4096,chunk['tokens'],1,1]
                coordinates=next(c['positions'] for c in case['chunks'] if c.get('image_positions')==image_index)
                assert len(coordinates)==chunk['tokens']
                span=next(s for s in position_fixture['spans'] if s['model']==key[0] and s['case']==key[1] and s['image_index']==image_index)
                assert span['start']==position and span['physical_tokens']==chunk['tokens'] and span['position_count']==chunk['position_count']
                assert span['decoder_axes']==[[p[0],p[2],p[1],p[3]] for p in coordinates]
                encoder=next(s for s in position_fixture['encoder_sources'] if s['model']==key[0] and s['case']==key[1])
                assert encoder['encoder_output_sha256']==case['encoder_output_sha256']
                feature_key=(key[0],images[image_index]['sha256'])
                digest=tensors[f'image_{image_index}_embeddings']['sha256']
                assert feature_key not in features or features[feature_key]==digest,'vision embeddings must be text-independent'
                features[feature_key]=digest
                total_spans+=1;image_index+=1
            physical+=chunk['tokens'];position+=chunk['position_count']
        assert image_index==len(images)==case['encoded']['images']
        assert physical==case['encoded']['physical_prompt_tokens']==case['decoded']['physical_prompt_tokens']
        assert position==case['encoded']['effective_positions']
        generation=case['generation']
        assert 1<=len(generation)<=8 and physical+len(generation)<=case['decoded']['context']
        assert case['decoded']['generated_token_ids']==[s['token_id'] for s in generation]
        for step,item in enumerate(generation):
            assert item['greedy_step']==step and (not item['eos'] or step==len(generation)-1)
            assert f'logits_{step}' in tensors
        assert case['output_bytes_hex']==''.join(s['piece_hex'] for s in generation)
        assert case['text']==bytes.fromhex(case['output_bytes_hex']).decode('utf8',errors='replace')
        if key[0]=='qwen2-vl-2b' and key[1] in ['left-red','left-blue']:
            previous=next(c for c in original_vision['cases'] if c['model']==key[0] and c['canonical']==images[0]['canonical'])
            for tensor in previous['tensors']:
                current='image_0_embeddings' if tensor['name']=='projected_embeddings' else 'vision_0_'+tensor['name']
                assert tensors[current]['sha256']==tensor['sha256'],'corrected build changed the Qwen vision graph'
        if key[0]=='llava-1.5-7b':
            comparison=case['independent_vision_comparison']
            assert comparison['source_sha256']==sha(ROOT/'resources/tool-assessment/llava_numpy.py')
            assert comparison['standard_math_source_sha256']==sha(ROOT/'resources/tool-assessment/scalar_math.c')
            assert len(comparison['comparisons'])==25
            assert comparison['tolerances']=={'absolute':0.003,'relative':0.001,'normalized_l2':0.0005}
            for item in comparison['comparisons']:
                assert item['outside_tolerance']==0 and item['normalized_l2']<=0.0005
                assert item['reference_sha256']==tensors[item['name']]['sha256']
            assert {int(n.removeprefix('vision_0_layer_out-')) for n in tensors if n.removeprefix('vision_0_layer_out-').isdigit()}==set(range(23))
    assert total_spans==10
    assert len(reference['image_sensitivity'])==3
    assert all(c['prompt_bytes_equal'] and c['maximum_absolute_logit_change']>0 and c['relative_l2_change']>0 for c in reference['image_sensitivity'])
    qwen='qwen2-vl-2b'
    for a,b in [('ordered-images','reversed-images'),('left-red','left-blue')]:
        logits=lambda key: next(t['sha256'] for t in cases[(qwen,key)]['tensors'] if t['name']=='logits_0')
        assert logits(a)!=logits(b),'conditioned logits must respond to image identity/order'
    assert any(item['eos'] for case in cases.values() for item in case['generation'])
    extra=json.loads((ROOT/'fixtures/vlm/qwen2-vision-extra-v1.json').read_text())
    assert extra['reference_revision']==reference['reference_revision']
    assert extra['wrapper_source_sha256']==sha(ROOT/'resources/tool-assessment/vision_oracle.cpp')
    assert extra['capture_source_sha256']==sha(ROOT/'resources/tool-assessment/capture_qwen_vision_extra.py')
    assert extra['native_execution_qualified'] is False
    vision=next(c for c in selected['qwen2-vl-2b']['components'] if c['role']=='vision-projector')
    assert extra['component_sha256']==vision['sha256']
    assert len(extra['cases'])==3
    for case in extra['cases']:
        assert case['canonical_sha256']==sha(ROOT/case['canonical'])
        assert len(case['tensors'])==33
        assert {t['name'] for t in case['tensors']}=={'projected_embeddings'} | {f'layer_out-{i}' for i in range(32)}
        for tensor in case['tensors']:
            assert math.prod(tensor['shape'])==tensor['count']>0
            assert len(tensor['sha256'])==64
            assert all(math.isfinite(tensor[field]) for field in ['sum','l2','minimum','maximum'])
            assert all(0<=sample['index']<tensor['count'] and math.isfinite(sample['value']) for sample in tensor['samples'])
    check_qwen_language(reference)
    print('PASS eight frozen image-conditioned cases, three additional vision geometries, ten image spans, independent corrected vision/language and unchanged tolerances; reference integrity does not qualify native execution')


if __name__=='__main__':main()
