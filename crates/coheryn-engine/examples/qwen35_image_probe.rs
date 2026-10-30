//! Native-only multimodal qualification runner using inert authored requests.
use coheryn_engine::{
    adapters::Execution,
    multimodal::{
        image::ImageLimits,
        qwen2vl::{ChatContent, ChatMessage, ChatPart},
    },
    planner::{HardwareSnapshot, Profile},
    qwen35::Model,
    *,
};
use serde_json::json;
use std::{fs, io::Write, path::PathBuf};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 3 {
        return Err(Error::request(
            "usage: qwen35_image_probe CHECKPOINT REQUEST OUTPUT",
        ));
    }
    let request: serde_json::Value =
        serde_json::from_slice(&fs::read(&a[1])?).map_err(|e| Error::request(e.to_string()))?;
    let directory = PathBuf::from(&a[2]);
    fs::create_dir_all(&directory)?;
    let cancel = Cancellation::default();
    let model = Model::load(
        &a[0],
        ResourceLimits {
            max_memory: 1 << 30,
            context: 256,
        },
        &cancel,
    )?;
    let images: Vec<Vec<u8>> = request["images"]
        .as_array()
        .ok_or_else(|| Error::request("images required"))?
        .iter()
        .map(|s| fs::read(s.as_str().unwrap()))
        .collect::<std::io::Result<_>>()?;
    let mut parts: Vec<_> = (0..images.len())
        .map(|index| ChatPart::Image { index })
        .collect();
    parts.push(ChatPart::Text {
        text: request["text"].as_str().unwrap().into(),
    });
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: ChatContent::Parts(parts),
    }];
    let prompt = model.render_images(&messages, images.len())?;
    let hardware = HardwareSnapshot {
        os: "probe".into(),
        architecture: "probe".into(),
        logical_cpus: 2,
        cpu_features: vec![],
        ram_total_bytes: None,
        ram_available_bytes: None,
        swap_total_bytes: None,
        swap_used_bytes: None,
        observed_unix_ms: 0,
        gpu_available_bytes: None,
        gpu_backend_qualified: false,
    };
    let prepared = model.prepare_images(
        &messages,
        images,
        ImageLimits::default(),
        GenerationOptions {
            max_tokens: 4,
            ..Default::default()
        },
        256,
        Execution {
            profile: Profile::Reference,
            threads: Some(1),
            hardware: &hardware,
        },
        1,
        &cancel,
    )?;
    let mut image = 0;
    let mut captures = Vec::new();
    let mut save = |name: String, data: &[f32]| -> Result<()> {
        let mut file = fs::File::create(directory.join(format!("{name}.f32")))?;
        for value in data {
            file.write_all(&value.to_le_bytes())?;
        }
        captures.push(json!({"name":name,"length":data.len()}));
        Ok(())
    };
    let mut state = model.begin_images(
        prepared,
        &cancel,
        None,
        Some(&mut |c| {
            save(format!("{image}-{}", c.name), c.values)?;
            if c.name == "merged" {
                image += 1;
            }
            Ok(())
        }),
    )?;
    let tokens = state.prompt_token_ids().to_vec();
    let teacher: Vec<u32> = request["teacher"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as u32)
        .collect();
    model.trace_images(&state, &teacher, &cancel, &mut |position, name, data| {
        if position + 1 >= tokens.len() && (position + 1 == tokens.len() || name == "logits") {
            save(format!("{position}-{name}"), data)?;
        }
        Ok(())
    })?;
    let mut events = Vec::new();
    while !state.is_done() {
        if state.needs_forward() {
            model.step_batch(&mut [&mut state], None)?;
        }
        if let Some(e) = state.sample(&model)? {
            events.push(e);
        }
    }
    let (result, cache) = model.finish_generation(state)?;
    assert!(cache.is_none());
    println!(
        "{}",
        json!({"prompt":prompt,"prompt_token_ids":tokens,"generation":result,"events":events,"after":model.memory(),"captures":captures})
    );
    Ok(())
}
