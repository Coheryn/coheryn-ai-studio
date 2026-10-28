//! Compatible model-token batches with shared bounded matrix weight reads.
use super::*;
#[derive(Debug, Default)]
pub struct BatchStep {
    pub width: usize,
    pub completed_tokens: usize,
    pub matrix_tiles: u64,
    pub matrix_bytes: u64,
}
struct Lane<'a> {
    token: u32,
    session: &'a mut Session,
    cancel: &'a Cancellation,
    error: &'a mut Option<Error>,
}
fn each(
    lanes: &mut [Lane<'_>],
    mut apply: impl FnMut(&mut Session, u32, &Cancellation) -> Result<()>,
) {
    for lane in lanes.iter_mut().filter(|l| l.error.is_none()) {
        if let Err(error) = lane
            .cancel
            .check()
            .and_then(|()| apply(lane.session, lane.token, lane.cancel))
        {
            *lane.error = Some(error);
        }
    }
}
impl Model {
    fn batch_matrix(
        &self,
        name: &str,
        lanes: &mut [Lane<'_>],
        select: fn(&mut Scratch) -> (&[f32], &mut [f32]),
        stats: &mut BatchStep,
    ) -> Result<()> {
        let tensor = self.tensor(name)?;
        let mut vectors = Vec::with_capacity(lanes.len());
        for lane in lanes.iter_mut() {
            let (input, output) = select(&mut lane.session.scratch);
            vectors.push(tensor::BatchVector {
                input,
                output,
                workspace: &mut lane.session.workspace,
                cancel: lane.cancel,
                error: lane.error,
            });
        }
        let io = tensor::matmul_batch(
            &self.storage,
            self.descriptor.data_offset,
            &tensor,
            &mut vectors,
        )?;
        stats.matrix_tiles += io.tiles;
        stats.matrix_bytes += io.bytes;
        Ok(())
    }
    /// One token per independent lane, sharing each matrix tile across the group.
    /// Model/worker/I/O contracts must match; positions, context lengths and decoder
    /// state may differ. Lane failures are observed by sample/next_token, not propagated
    /// into healthy lanes. A shared model-file failure invalidates all participants.
    pub fn step_batch(&self, states: &mut [&mut GenerationState]) -> Result<BatchStep> {
        if states.is_empty() || states.len() > 16 {
            return Err(Error::request("batch width must be in 1..=16"));
        }
        for state in states.iter() {
            if !Weak::ptr_eq(&state.model, &Arc::downgrade(&self.cache_owner))
                || !state.compatible(states[0])
            {
                return Err(Error::request("incompatible native generation batch"));
            }
        }
        let start = Instant::now();
        let mut prefilling = Vec::with_capacity(states.len());
        let mut lanes = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            prefilling.push(state.is_prefilling());
            let token = match state.next_token() {
                Ok(token) => token,
                Err(error) => {
                    state.error = Some(error);
                    0
                }
            };
            lanes.push(Lane {
                token,
                session: state.session.as_mut().expect("live session"),
                cancel: &state.cancel,
                error: &mut state.error,
            });
        }
        let mut stats = BatchStep {
            width: lanes.iter().filter(|lane| lane.error.is_none()).count(),
            ..Default::default()
        };
        let result = self.forward_batch(&mut lanes, &mut stats);
        if let Err(error) = result {
            for lane in &mut lanes {
                if lane.error.is_none() {
                    *lane.error = Some(error.clone());
                }
            }
        }
        drop(lanes);
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        for (state, prefilling) in states.iter_mut().zip(prefilling) {
            state.batching.strategy = "continuous-weight-tiled".into();
            if state.error.is_none() {
                stats.completed_tokens += 1;
                state.forwarded(elapsed, prefilling, stats.width);
            }
        }
        Ok(stats)
    }
    fn forward_batch(&self, lanes: &mut [Lane<'_>], stats: &mut BatchStep) -> Result<()> {
        let c = &self.config;
        each(lanes, |session, token, _| {
            if token as usize >= c.vocabulary {
                return Err(Error::request("token ID outside vocabulary"));
            }
            if session.position >= session.context {
                return Err(Error::resource("context capacity exhausted"));
            }
            self.vector(
                "token_embd.weight",
                token as usize,
                &mut session.scratch.x,
                &mut session.workspace,
            )
        });
        for layer in 0..c.layers {
            let prefix = format!("blk.{layer}");
            each(lanes, |session, _, _| {
                let s = &mut session.scratch;
                self.vector(
                    &format!("{prefix}.attn_norm.weight"),
                    0,
                    &mut s.norm_weight,
                    &mut session.workspace,
                )?;
                cpu::rms_norm(&s.x, &s.norm_weight, c.norm_epsilon, &mut s.normalized);
                Ok(())
            });
            self.batch_matrix(
                &format!("{prefix}.attn_q.weight"),
                lanes,
                |s| (&s.normalized, &mut s.q),
                stats,
            )?;
            self.batch_matrix(
                &format!("{prefix}.attn_k.weight"),
                lanes,
                |s| (&s.normalized, &mut s.k),
                stats,
            )?;
            self.batch_matrix(
                &format!("{prefix}.attn_v.weight"),
                lanes,
                |s| (&s.normalized, &mut s.v),
                stats,
            )?;
            each(lanes, |session, _, cancel| {
                let s = &mut session.scratch;
                for (name, out) in [
                    ("attn_q", &mut s.q),
                    ("attn_k", &mut s.k),
                    ("attn_v", &mut s.v),
                ] {
                    self.vector(
                        &format!("{prefix}.{name}.bias"),
                        0,
                        &mut s.bias[..out.len()],
                        &mut session.workspace,
                    )?;
                    for (x, b) in out.iter_mut().zip(&s.bias) {
                        *x += b;
                    }
                }
                cpu::rope(&mut s.q, c.head_dim, session.position, c.rope_base);
                cpu::rope(&mut s.k, c.head_dim, session.position, c.rope_base);
                session.keys[layer].append(&s.k, session.context, &mut session.buffers)?;
                session.values[layer].append(&s.v, session.context, &mut session.buffers)?;
                cpu::attention_rows(
                    &s.q,
                    &session.keys[layer].data,
                    &session.values[layer].data,
                    c.kv_heads,
                    c.head_dim,
                    cpu::AttentionWorkspace {
                        output: &mut s.attention,
                        scores: &mut s.scores,
                        accumulator: &mut s.attention_accumulator,
                    },
                    || cancel.check(),
                )
            });
            self.batch_matrix(
                &format!("{prefix}.attn_output.weight"),
                lanes,
                |s| (&s.attention, &mut s.projection),
                stats,
            )?;
            each(lanes, |session, _, _| {
                let s = &mut session.scratch;
                for (x, p) in s.x.iter_mut().zip(&s.projection) {
                    *x += p;
                }
                self.vector(
                    &format!("{prefix}.ffn_norm.weight"),
                    0,
                    &mut s.norm_weight,
                    &mut session.workspace,
                )?;
                cpu::rms_norm(&s.x, &s.norm_weight, c.norm_epsilon, &mut s.normalized);
                Ok(())
            });
            self.batch_matrix(
                &format!("{prefix}.ffn_gate.weight"),
                lanes,
                |s| (&s.normalized, &mut s.gate),
                stats,
            )?;
            self.batch_matrix(
                &format!("{prefix}.ffn_up.weight"),
                lanes,
                |s| (&s.normalized, &mut s.up),
                stats,
            )?;
            each(lanes, |session, _, _| {
                let s = &mut session.scratch;
                cpu::swiglu(&mut s.gate, &s.up);
                Ok(())
            });
            self.batch_matrix(
                &format!("{prefix}.ffn_down.weight"),
                lanes,
                |s| (&s.gate, &mut s.projection),
                stats,
            )?;
            each(lanes, |session, _, _| {
                let s = &mut session.scratch;
                for (x, p) in s.x.iter_mut().zip(&s.projection) {
                    *x += p;
                }
                Ok(())
            });
        }
        each(lanes, |session, _, _| {
            let s = &mut session.scratch;
            self.vector(
                "output_norm.weight",
                0,
                &mut s.norm_weight,
                &mut session.workspace,
            )?;
            cpu::rms_norm(&s.x, &s.norm_weight, c.norm_epsilon, &mut s.normalized);
            Ok(())
        });
        let output = if self.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        self.batch_matrix(output, lanes, |s| (&s.normalized, &mut s.logits), stats)?;
        each(lanes, |session, token, _| {
            session.position += 1;
            session.tokens.push(token);
            Ok(())
        });
        Ok(())
    }
}
