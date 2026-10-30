//! Exact-prefix hybrid state. No rewind, cross-model reuse, or serialized tensors.
use super::*;
use execution::Snapshot;
use std::sync::{Arc, Weak};

pub struct HybridCache {
    pub(super) snapshot: Snapshot,
    pub(super) tokens: Vec<u32>,
    owner: Weak<()>,
    context: usize,
    threads: usize,
    _buffers: crate::memory::BufferAccounting,
    reservation: Reservation,
}
impl HybridCache {
    pub fn is_live(&self) -> bool {
        self.owner.strong_count() > 0
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.reservation.bytes()
    }
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
    fn matches(&self, model: &Model, context: usize, threads: usize) -> bool {
        Weak::ptr_eq(&self.owner, &Arc::downgrade(&model.owner))
            && self.context == context
            && self.threads == threads
            && self
                .reservation
                .covers(&model.budget, self.reservation.bytes())
    }
}
pub(super) enum CacheSlot {
    Existing(HybridCache),
    Fresh(Reservation),
}
impl CacheSlot {
    pub(super) fn bytes(&self) -> u64 {
        match self {
            Self::Existing(c) => c.reserved_bytes(),
            Self::Fresh(r) => r.bytes(),
        }
    }
    pub(super) fn initialize(
        self,
        model: &Model,
        context: usize,
        threads: usize,
    ) -> Result<HybridCache> {
        match self {
            Self::Existing(c) => Ok(c),
            Self::Fresh(reservation) => {
                let snapshot = Snapshot::new(&model.config, context)?;
                let mut tokens = Vec::new();
                tokens
                    .try_reserve_exact(context)
                    .map_err(|_| Error::resource("cache token ledger allocation failed"))?;
                let mut buffers = model.budget.track_buffers(reservation.bytes());
                buffers.grow(snapshot.capacity_bytes() + tokens.capacity() as u64 * 4)?;
                Ok(HybridCache {
                    snapshot,
                    tokens,
                    owner: Arc::downgrade(&model.owner),
                    context,
                    threads,
                    _buffers: buffers,
                    reservation,
                })
            }
        }
    }
}
impl Model {
    pub fn cache_bytes(&self, context: usize) -> Result<u64> {
        if context == 0 || context > self.limits.context {
            return Err(Error::request("cache context exceeds loaded capacity"));
        }
        Snapshot::bytes(&self.config, context)
    }
    pub fn prepare_cached_generation(
        &self,
        options: GenerationOptions,
        context: usize,
        execution: crate::adapters::Execution<'_>,
        max_concurrency: usize,
        mut cache: Option<HybridCache>,
        limit: u64,
    ) -> Result<PreparedGeneration> {
        let threads = crate::planner::execution_threads(
            execution.profile,
            execution.threads,
            execution.hardware.logical_cpus,
            max_concurrency,
        )?;
        let bytes = self.cache_bytes(context)?;
        if cache
            .as_ref()
            .is_some_and(|c| !c.matches(self, context, threads) || c.reserved_bytes() > limit)
        {
            cache = None;
        }
        let run = || crate::adapters::Execution {
            profile: execution.profile,
            threads: execution.threads,
            hardware: execution.hardware,
        };
        let mut prepared =
            match self.prepare_generation(options.clone(), context, run(), max_concurrency) {
                Err(e) if e.code == ErrorCode::ResourceExhausted && cache.is_some() => {
                    drop(cache.take());
                    self.prepare_generation(options, context, run(), max_concurrency)?
                }
                r => r?,
            };
        prepared.cache = if let Some(cache) = cache {
            Some(CacheSlot::Existing(cache))
        } else if bytes <= limit {
            self.budget.reserve(bytes).ok().map(CacheSlot::Fresh)
        } else {
            None
        };
        Ok(prepared)
    }
    pub(super) fn cache_boundary(&self, prompt: &str, tokens: &[u32]) -> Result<usize> {
        let end = prompt.rfind("<|im_start|>assistant\n").unwrap_or(0);
        if end == 0 {
            return Ok(tokens.len().saturating_sub(1));
        }
        let prefix = self.tokenize(&prompt[..end])?;
        Ok(
            if tokens.starts_with(&prefix) && prefix.len() < tokens.len() {
                prefix.len()
            } else {
                0
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the separately acquired pinned Qwen3.5 checkpoint"]
    fn hybrid_cache_matches_fresh_conversation_and_reclaims_on_failure() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = Model::load(
            root.join(".cache/qwen3.5-0.8b-source"),
            ResourceLimits {
                context: 128,
                max_memory: 1 << 30,
            },
            &Default::default(),
        )
        .unwrap();
        let hardware = crate::planner::HardwareSnapshot {
            os: "test".into(),
            architecture: "test".into(),
            logical_cpus: 1,
            cpu_features: vec![],
            ram_total_bytes: None,
            ram_available_bytes: None,
            swap_total_bytes: None,
            swap_used_bytes: None,
            observed_unix_ms: 0,
            gpu_available_bytes: None,
            gpu_backend_qualified: false,
        };
        let execution = || crate::adapters::Execution {
            profile: crate::planner::Profile::Reference,
            threads: Some(1),
            hardware: &hardware,
        };
        let options = GenerationOptions {
            max_tokens: 2,
            ..Default::default()
        };
        let mut messages = vec![Message {
            role: "user".into(),
            content: "Say hello.".into(),
            ..Default::default()
        }];
        let prompt = model.render(&messages).unwrap();
        let prepared = model
            .prepare_cached_generation(options.clone(), 128, execution(), 1, None, 64 << 20)
            .unwrap();
        let (first, cache) = model
            .generate_prepared_cached(
                &prompt,
                prepared,
                &Default::default(),
                &mut || Ok(()),
                &mut |_| Ok(()),
            )
            .unwrap();
        let cache = cache.expect("prompt checkpoint");
        assert!(cache.token_count() > 0);
        assert!(cache.reserved_bytes() <= 64 << 20);
        messages.push(Message {
            role: "assistant".into(),
            content: first.text,
            ..Default::default()
        });
        messages.push(Message {
            role: "user".into(),
            content: "你好 again.".into(),
            ..Default::default()
        });
        let prompt = model.render(&messages).unwrap();
        let prepared = model
            .prepare_cached_generation(options.clone(), 128, execution(), 1, Some(cache), 64 << 20)
            .unwrap();
        let (cached, cache) = model
            .generate_prepared_cached(
                &prompt,
                prepared,
                &Default::default(),
                &mut || Ok(()),
                &mut |_| Ok(()),
            )
            .unwrap();
        assert!(cached.reused_prompt_tokens > 0);
        let fresh = model
            .generate(&prompt, &options, 1, &Default::default(), &mut |_| Ok(()))
            .unwrap();
        assert_eq!(cached.token_ids, fresh.token_ids);
        assert_eq!(cached.text, fresh.text);
        assert_eq!(cached.prompt_tokens, fresh.prompt_tokens);
        let prepared = model
            .prepare_cached_generation(options, 128, execution(), 1, cache, 64 << 20)
            .unwrap();
        let error = model
            .generate_prepared_cached(
                &prompt,
                prepared,
                &Default::default(),
                &mut || Ok(()),
                &mut |_| {
                    Err(Error::new(
                        ErrorCode::Cancelled,
                        "test consumer cancellation",
                    ))
                },
            )
            .err()
            .unwrap();
        assert_eq!(error.code, ErrorCode::Cancelled);
        assert_eq!(model.memory().reserved, RESIDENT_METADATA_BYTES);
        assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    }
}
