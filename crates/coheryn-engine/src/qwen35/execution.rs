use super::{
    Config,
    kernels::{self, ConvState, DeltaState},
};
use crate::{
    Cancellation, Result, cpu, memory,
    safetensors::{BoundTensor, TensorSet},
    tensor::{self, Workspace},
};

type LayerObserver<'a> = dyn FnMut(&str, &[f32]) -> Result<()> + 'a;

pub(crate) struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    gate: BoundTensor,
    up: BoundTensor,
    down: BoundTensor,
    mixer: Mixer,
}
enum Mixer {
    Full {
        q: BoundTensor,
        k: BoundTensor,
        v: BoundTensor,
        out: BoundTensor,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
    Linear {
        qkv: BoundTensor,
        z: BoundTensor,
        a: BoundTensor,
        b: BoundTensor,
        out: BoundTensor,
        conv: Vec<f32>,
        a_log: Vec<f32>,
        dt: Vec<f32>,
        norm: Vec<f32>,
    },
}
enum MixerState {
    Full { keys: Vec<f32>, values: Vec<f32> },
    Linear { conv: ConvState, delta: DeltaState },
}
pub(crate) struct Graph {
    layers: Vec<Layer>,
    embedding: BoundTensor,
    output: BoundTensor,
    norm: Vec<f32>,
}
pub(crate) struct Session {
    layers: Vec<MixerState>,
    pub(crate) logits: Vec<f32>,
    work: Workspace,
    hidden: Vec<f32>,
    normalized: Vec<f32>,
    mixed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    z: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    delta_q: Vec<f32>,
    delta_k: Vec<f32>,
    attention: Vec<f32>,
    norm_head: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    projected: Vec<f32>,
    final_norm: Vec<f32>,
    pub(crate) position: usize,
    pub(super) rotary: Option<[usize; 3]>,
    capacity: usize,
}
pub(super) struct Snapshot {
    layers: Vec<MixerState>,
    position: usize,
    capacity: usize,
}
impl Snapshot {
    pub(super) fn bytes(c: &Config, context: usize) -> Result<u64> {
        let full = c.layers.iter().filter(|v| **v).count();
        let linear = c.layers.len() - full;
        let kv = memory::product(&[full, context, c.kv_heads, c.head_dim, 2])?;
        let delta = memory::product(&[linear, c.linear_value_heads, c.key_dim, c.value_dim])?;
        let conv = memory::product(&[
            linear,
            2 * c.linear_key_heads * c.key_dim + c.linear_value_heads * c.value_dim,
            c.conv_kernel,
        ])?;
        (kv as u64)
            .checked_add(delta as u64)
            .and_then(|v| v.checked_add(conv as u64))
            .and_then(|v| v.checked_mul(4))
            .and_then(|v| {
                v.checked_add(
                    context as u64 * 4
                        + 8192
                        + (c.layers.len().next_power_of_two() * std::mem::size_of::<MixerState>())
                            as u64,
                )
            })
            .ok_or_else(|| crate::Error::resource("hybrid cache byte overflow"))
    }
    pub(super) fn new(c: &Config, context: usize) -> Result<Self> {
        Ok(Self {
            layers: Session::states(c, context)?,
            position: 0,
            capacity: context,
        })
    }
    pub(super) fn capacity_bytes(&self) -> u64 {
        (self.layers.capacity() * std::mem::size_of::<MixerState>()
            + self
                .layers
                .iter()
                .map(|s| match s {
                    MixerState::Full { keys, values } => (keys.capacity() + values.capacity()) * 4,
                    MixerState::Linear { conv, delta } => {
                        conv.capacity_bytes() + delta.capacity_bytes()
                    }
                })
                .sum::<usize>()) as u64
    }
    pub(super) fn capture(&mut self, session: &Session) -> Result<()> {
        if self.capacity != session.capacity {
            return Err(crate::Error::request("hybrid cache context mismatch"));
        }
        copy_states(&mut self.layers, &session.layers)?;
        self.position = session.position;
        Ok(())
    }
    pub(super) fn restore(&self, session: &mut Session) -> Result<()> {
        if self.capacity != session.capacity {
            return Err(crate::Error::request("hybrid cache context mismatch"));
        }
        copy_states(&mut session.layers, &self.layers)?;
        session.position = self.position;
        Ok(())
    }
}
fn copy_states(destination: &mut [MixerState], source: &[MixerState]) -> Result<()> {
    if destination.len() != source.len() {
        return Err(crate::Error::request("hybrid cache layer mismatch"));
    }
    for (to, from) in destination.iter_mut().zip(source) {
        match (to, from) {
            (MixerState::Full { keys: a, values: b }, MixerState::Full { keys: x, values: y })
                if a.len() == x.len() && b.len() == y.len() =>
            {
                a.copy_from_slice(x);
                b.copy_from_slice(y);
            }
            (
                MixerState::Linear { conv: a, delta: b },
                MixerState::Linear { conv: x, delta: y },
            ) => {
                a.copy_from(x)?;
                b.copy_from(y)?;
            }
            _ => return Err(crate::Error::request("hybrid cache mixer mismatch")),
        }
    }
    Ok(())
}
impl Graph {
    pub(crate) fn bind(set: &TensorSet, c: &Config) -> Result<Self> {
        let mut layers = Vec::new();
        for (i, full) in c.layers.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            let mixer = if *full {
                let p = format!("{p}.self_attn");
                Mixer::Full {
                    q: set.bind(&format!("{p}.q_proj.weight"))?,
                    k: set.bind(&format!("{p}.k_proj.weight"))?,
                    v: set.bind(&format!("{p}.v_proj.weight"))?,
                    out: set.bind(&format!("{p}.o_proj.weight"))?,
                    q_norm: set.read_vector(&format!("{p}.q_norm.weight"))?,
                    k_norm: set.read_vector(&format!("{p}.k_norm.weight"))?,
                }
            } else {
                let p = format!("{p}.linear_attn");
                Mixer::Linear {
                    qkv: set.bind(&format!("{p}.in_proj_qkv.weight"))?,
                    z: set.bind(&format!("{p}.in_proj_z.weight"))?,
                    a: set.bind(&format!("{p}.in_proj_a.weight"))?,
                    b: set.bind(&format!("{p}.in_proj_b.weight"))?,
                    out: set.bind(&format!("{p}.out_proj.weight"))?,
                    conv: set.read_vector(&format!("{p}.conv1d.weight"))?,
                    a_log: set.read_vector(&format!("{p}.A_log"))?,
                    dt: set.read_vector(&format!("{p}.dt_bias"))?,
                    norm: set.read_vector(&format!("{p}.norm.weight"))?,
                }
            };
            layers.push(Layer {
                input_norm: set.read_vector(&format!("{p}.input_layernorm.weight"))?,
                post_norm: set.read_vector(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: set.bind(&format!("{p}.mlp.gate_proj.weight"))?,
                up: set.bind(&format!("{p}.mlp.up_proj.weight"))?,
                down: set.bind(&format!("{p}.mlp.down_proj.weight"))?,
                mixer,
            });
        }
        Ok(Self {
            layers,
            embedding: set.bind("model.language_model.embed_tokens.weight")?,
            output: set.bind(if c.tied {
                "model.language_model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            })?,
            norm: set.read_vector("model.language_model.norm.weight")?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        set: &TensorSet,
        c: &Config,
        token: u32,
        s: &mut Session,
        logits: bool,
        cancel: &Cancellation,
        observe: Option<&mut LayerObserver<'_>>,
        cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<()> {
        self.forward_input(set, c, token, None, s, logits, cancel, observe, cooperate)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_input(
        &self,
        set: &TensorSet,
        c: &Config,
        token: u32,
        embedding: Option<&[f32]>,
        s: &mut Session,
        logits: bool,
        cancel: &Cancellation,
        mut observe: Option<&mut LayerObserver<'_>>,
        mut cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<()> {
        let mut error = None;
        let mut lanes = [Lane {
            session: s,
            token,
            embedding,
            logits,
            cancel,
            error: &mut error,
        }];
        self.forward_batch(set, c, &mut lanes, &mut observe, &mut cooperate)?;
        error.map_or(Ok(()), Err)
    }
    pub(super) fn forward_batch(
        &self,
        set: &TensorSet,
        c: &Config,
        lanes: &mut [Lane<'_>],
        observe: &mut Option<&mut LayerObserver<'_>>,
        cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<crate::inference::BatchStep> {
        if lanes.is_empty() || lanes.len() > 16 {
            return Err(crate::Error::request("hybrid batch width must be 1..=16"));
        }
        set.unchanged()?;
        let mut stats = crate::inference::BatchStep {
            width: lanes.len(),
            ..Default::default()
        };
        for lane in lanes.iter_mut().filter(|l| l.error.is_none()) {
            let s = &mut lane.session;
            let result = (|| {
                lane.cancel.check()?;
                if s.position >= s.capacity || lane.token as usize >= c.vocabulary {
                    return Err(crate::Error::request(
                        "hybrid token/context boundary exceeded",
                    ));
                }
                if let Some(embedding) = lane.embedding {
                    if embedding.len() != c.hidden {
                        return Err(crate::Error::request("invalid visual embedding width"));
                    }
                    s.hidden.copy_from_slice(embedding);
                    Ok(())
                } else {
                    let (storage, tensor) = set.bound(&self.embedding)?;
                    tensor::read_row(
                        storage,
                        0,
                        tensor,
                        lane.token as usize,
                        &mut s.hidden,
                        &mut s.work,
                    )
                    .map_err(|error| set.source_error(error))
                }
            })();
            if let Err(e) = result {
                *lane.error = Some(e);
            }
        }
        stats.width = lanes.iter().filter(|l| l.error.is_none()).count();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(y) = cooperate.as_deref_mut() {
                y()?;
            }
            each(lanes, |s, _, _| {
                kernels::norm(
                    &s.hidden,
                    &layer.input_norm,
                    c.epsilon,
                    c.zero_centered_norms,
                    &mut s.normalized,
                )
            });
            match &layer.mixer {
                Mixer::Linear {
                    qkv,
                    z,
                    a,
                    b,
                    out,
                    conv,
                    a_log,
                    dt,
                    norm,
                } => {
                    let keys = c.linear_key_heads * c.key_dim;
                    let values = c.linear_value_heads * c.value_dim;
                    let width = 2 * keys + values;
                    matrix(
                        set,
                        qkv,
                        lanes,
                        |s| (&s.normalized, &mut s.mixed[..width], &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                    matrix(
                        set,
                        z,
                        lanes,
                        |s| (&s.normalized, &mut s.z[..values], &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                    matrix(
                        set,
                        a,
                        lanes,
                        |s| (&s.normalized, &mut s.a, &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                    matrix(
                        set,
                        b,
                        lanes,
                        |s| (&s.normalized, &mut s.b, &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                    each(lanes, |s, _, cancel| {
                        let MixerState::Linear {
                            conv: history,
                            delta,
                        } = &mut s.layers[i]
                        else {
                            return Err(crate::Error::request("hybrid mixer mismatch"));
                        };
                        history.step(&s.mixed[..width], conv, &mut s.q[..width], cancel)?;
                        let repeat = c.linear_value_heads / c.linear_key_heads;
                        for head in 0..c.linear_value_heads {
                            let source = (head / repeat) * c.key_dim;
                            let destination = head * c.key_dim;
                            s.delta_q[destination..destination + c.key_dim]
                                .copy_from_slice(&s.q[source..source + c.key_dim]);
                            s.delta_k[destination..destination + c.key_dim]
                                .copy_from_slice(&s.q[keys + source..keys + source + c.key_dim]);
                            s.a[head] = (if c.ssm_pre_exponentiated {
                                a_log[head]
                            } else {
                                -a_log[head].exp()
                            }) * kernels::softplus(s.a[head] + dt[head]);
                            s.b[head] = kernels::sigmoid(s.b[head]);
                        }
                        delta.step(
                            &s.delta_q,
                            &s.delta_k,
                            &s.q[2 * keys..width],
                            &s.a,
                            &s.b,
                            &mut s.attention[..values],
                            cancel,
                        )?;
                        for (head, ((row, gate), target)) in s.attention[..values]
                            .chunks_exact(c.value_dim)
                            .zip(s.z[..values].chunks_exact(c.value_dim))
                            .zip(s.v[..values].chunks_exact_mut(c.value_dim))
                            .enumerate()
                        {
                            if head % 8 == 0 {
                                cancel.check()?;
                            }
                            kernels::norm(row, norm, c.epsilon, false, target)?;
                            for (value, g) in target.iter_mut().zip(gate) {
                                *value *= kernels::silu(*g);
                            }
                        }

                        Ok(())
                    });
                    matrix(
                        set,
                        out,
                        lanes,
                        |s| (&s.v[..values], &mut s.projected, &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                }
                Mixer::Full {
                    q,
                    k,
                    v,
                    out,
                    q_norm,
                    k_norm,
                } => {
                    let query_width = c.heads * c.head_dim;
                    let kv_width = c.kv_heads * c.head_dim;
                    matrix(
                        set,
                        q,
                        lanes,
                        |s| (&s.normalized, &mut s.mixed[..2 * query_width], &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                    each(lanes, |s, _, _| {
                        for h in 0..c.heads {
                            let base = h * 2 * c.head_dim;
                            kernels::norm(
                                &s.mixed[base..base + c.head_dim],
                                q_norm,
                                c.epsilon,
                                c.zero_centered_norms,
                                &mut s.q[h * c.head_dim..(h + 1) * c.head_dim],
                            )?;
                            s.z[h * c.head_dim..(h + 1) * c.head_dim].copy_from_slice(
                                &s.mixed[base + c.head_dim..base + 2 * c.head_dim],
                            );
                        }

                        Ok(())
                    });
                    matrix(
                        set,
                        k,
                        lanes,
                        |s| (&s.normalized, &mut s.k[..kv_width], &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                    matrix(
                        set,
                        v,
                        lanes,
                        |s| (&s.normalized, &mut s.v[..kv_width], &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                    each(lanes, |s, _, cancel| {
                        let MixerState::Full { keys, values } = &mut s.layers[i] else {
                            return Err(crate::Error::request("hybrid mixer mismatch"));
                        };
                        for row in s.k[..kv_width].chunks_exact_mut(c.head_dim) {
                            kernels::norm(
                                row,
                                k_norm,
                                c.epsilon,
                                c.zero_centered_norms,
                                &mut s.norm_head[..c.head_dim],
                            )?;
                            row.copy_from_slice(&s.norm_head[..c.head_dim]);
                        }
                        kernels::partial_mrope(
                            &mut s.q[..query_width],
                            c.head_dim,
                            c.rotary_dim,
                            s.rotary.unwrap_or([s.position; 3]),
                            c.rope_base,
                        )?;
                        kernels::partial_mrope(
                            &mut s.k[..kv_width],
                            c.head_dim,
                            c.rotary_dim,
                            s.rotary.unwrap_or([s.position; 3]),
                            c.rope_base,
                        )?;
                        let start = s.position * kv_width;
                        keys[start..start + kv_width].copy_from_slice(&s.k[..kv_width]);
                        values[start..start + kv_width].copy_from_slice(&s.v[..kv_width]);
                        cpu::attention(
                            &s.q[..query_width],
                            &keys[..start + kv_width],
                            &values[..start + kv_width],
                            c.kv_heads,
                            c.head_dim,
                            &mut s.attention[..query_width],
                            || cancel.check(),
                        )?;
                        for (value, gate) in s.attention[..query_width].iter_mut().zip(&s.z) {
                            *value *= kernels::sigmoid(*gate);
                        }

                        Ok(())
                    });
                    matrix(
                        set,
                        out,
                        lanes,
                        |s| (&s.attention[..query_width], &mut s.projected, &mut s.work),
                        &mut stats,
                        cooperate,
                    )?;
                }
            }
            each(lanes, |s, _, _| {
                for (x, update) in s.hidden.iter_mut().zip(&s.projected) {
                    *x += update;
                }
                kernels::norm(
                    &s.hidden,
                    &layer.post_norm,
                    c.epsilon,
                    c.zero_centered_norms,
                    &mut s.normalized,
                )
            });
            matrix(
                set,
                &layer.gate,
                lanes,
                |s| (&s.normalized, &mut s.gate, &mut s.work),
                &mut stats,
                cooperate,
            )?;
            matrix(
                set,
                &layer.up,
                lanes,
                |s| (&s.normalized, &mut s.up, &mut s.work),
                &mut stats,
                cooperate,
            )?;
            each(lanes, |s, _, _| {
                cpu::swiglu(&mut s.gate, &s.up);
                Ok(())
            });
            matrix(
                set,
                &layer.down,
                lanes,
                |s| (&s.gate, &mut s.projected, &mut s.work),
                &mut stats,
                cooperate,
            )?;
            each(lanes, |s, _, _| {
                for (x, update) in s.hidden.iter_mut().zip(&s.projected) {
                    *x += update;
                }
                Ok(())
            });
            if lanes.len() == 1
                && lanes[0].error.is_none()
                && let Some(observer) = observe.as_deref_mut()
            {
                observer(&format!("layer.{i}"), &lanes[0].session.hidden)?;
            }
        }
        each(lanes, |s, _, _| {
            kernels::norm(
                &s.hidden,
                &self.norm,
                c.epsilon,
                c.zero_centered_norms,
                &mut s.final_norm,
            )
        });
        // Non-final prefill rows avoid the vocabulary projection.
        matrix_selected(
            set,
            &self.output,
            lanes,
            |s| (&s.final_norm, &mut s.logits, &mut s.work),
            &mut stats,
            cooperate,
            true,
        )?;
        if lanes.len() == 1
            && lanes[0].logits
            && lanes[0].error.is_none()
            && let Some(observer) = observe.as_deref_mut()
        {
            observer("logits", &lanes[0].session.logits)?;
        }
        each(lanes, |s, _, _| {
            s.position += 1;
            Ok(())
        });
        stats.completed_tokens = lanes.iter().filter(|l| l.error.is_none()).count();
        Ok(stats)
    }
}

pub(super) struct Lane<'a> {
    pub token: u32,
    pub embedding: Option<&'a [f32]>,
    pub logits: bool,
    pub session: &'a mut Session,
    pub cancel: &'a Cancellation,
    pub error: &'a mut Option<crate::Error>,
}
fn each(
    lanes: &mut [Lane<'_>],
    mut action: impl FnMut(&mut Session, u32, &Cancellation) -> Result<()>,
) {
    for lane in lanes.iter_mut().filter(|l| l.error.is_none()) {
        if let Err(e) = lane
            .cancel
            .check()
            .and_then(|()| action(lane.session, lane.token, lane.cancel))
        {
            *lane.error = Some(e);
        }
    }
}
fn matrix(
    set: &TensorSet,
    binding: &BoundTensor,
    lanes: &mut [Lane<'_>],
    select: impl Fn(&mut Session) -> (&[f32], &mut [f32], &mut Workspace),
    stats: &mut crate::inference::BatchStep,
    cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
) -> Result<()> {
    matrix_selected(set, binding, lanes, select, stats, cooperate, false)
}
#[allow(clippy::too_many_arguments)]
fn matrix_selected(
    set: &TensorSet,
    binding: &BoundTensor,
    lanes: &mut [Lane<'_>],
    select: impl Fn(&mut Session) -> (&[f32], &mut [f32], &mut Workspace),
    stats: &mut crate::inference::BatchStep,
    cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
    logits_only: bool,
) -> Result<()> {
    let mut vectors = Vec::with_capacity(lanes.len());
    for lane in lanes
        .iter_mut()
        .filter(|l| l.error.is_none() && (!logits_only || l.logits))
    {
        let (input, output, workspace) = select(lane.session);
        vectors.push(tensor::BatchVector {
            input,
            output,
            workspace,
            cancel: lane.cancel,
            error: lane.error,
        });
    }
    if vectors.is_empty() {
        return Ok(());
    }
    let (storage, tensor) = set.bound(binding)?;
    let io = tensor::matmul_batch_cooperative(storage, 0, tensor, &mut vectors, cooperate)
        .map_err(|error| set.source_error(error))?;
    stats.matrix_tiles += io.tiles;
    stats.matrix_bytes += io.bytes;
    Ok(())
}

impl Session {
    pub(super) fn context_capacity(&self) -> usize {
        self.capacity
    }
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let vectors = [
            &self.logits,
            &self.hidden,
            &self.normalized,
            &self.mixed,
            &self.q,
            &self.k,
            &self.v,
            &self.z,
            &self.a,
            &self.b,
            &self.delta_q,
            &self.delta_k,
            &self.attention,
            &self.norm_head,
            &self.gate,
            &self.up,
            &self.projected,
            &self.final_norm,
        ]
        .iter()
        .map(|v| v.capacity() * std::mem::size_of::<f32>())
        .sum::<usize>();
        let states = self
            .layers
            .iter()
            .map(|s| match s {
                MixerState::Full { keys, values } => {
                    (keys.capacity() + values.capacity()) * std::mem::size_of::<f32>()
                }
                MixerState::Linear { conv, delta } => {
                    conv.capacity_bytes() + delta.capacity_bytes()
                }
            })
            .sum::<usize>();
        (vectors
            + states
            + self.work.capacity_bytes()
            + self.layers.capacity() * std::mem::size_of::<MixerState>()) as u64
    }
    pub(crate) fn bytes(c: &Config, context: usize, threads: usize) -> Result<u64> {
        let kv = memory::product(&[
            c.layers.iter().filter(|x| **x).count(),
            context,
            c.kv_heads,
            c.head_dim,
            2,
        ])?;
        let recurrent = memory::product(&[
            c.layers.iter().filter(|x| !**x).count(),
            c.linear_value_heads,
            c.key_dim,
            c.value_dim,
        ])?;
        let conv = memory::product(&[
            c.layers.iter().filter(|x| !**x).count(),
            2 * c.linear_key_heads * c.key_dim + c.linear_value_heads * c.value_dim,
            c.conv_kernel,
        ])?;
        let vectors = memory::product(&[
            c.hidden
                .max(c.intermediate)
                .max(2 * c.linear_key_heads * c.key_dim + c.linear_value_heads * c.value_dim)
                .max(2 * c.heads * c.head_dim)
                .max(c.linear_value_heads * c.key_dim),
            32,
        ])?
        .checked_add(c.vocabulary)
        .ok_or_else(|| crate::Error::resource("hybrid workspace overflow"))?;
        let count = kv
            .checked_add(recurrent)
            .and_then(|n| n.checked_add(conv))
            .and_then(|n| n.checked_add(vectors))
            .ok_or_else(|| crate::Error::resource("hybrid reservation overflow"))?;
        u64::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add((8 << 20) + memory::worker_bytes(threads).ok()? as u64))
            .ok_or_else(|| crate::Error::resource("hybrid byte reservation overflow"))
    }
    fn states(c: &Config, context: usize) -> Result<Vec<MixerState>> {
        let keys = c.linear_key_heads * c.key_dim;
        let values = c.linear_value_heads * c.value_dim;
        let channels = 2 * keys + values;
        let mut layers = Vec::new();
        for full in &c.layers {
            layers.push(if *full {
                MixerState::Full {
                    keys: memory::zeros(memory::product(&[context, c.kv_heads, c.head_dim])?)?,
                    values: memory::zeros(memory::product(&[context, c.kv_heads, c.head_dim])?)?,
                }
            } else {
                MixerState::Linear {
                    conv: ConvState::new(channels, c.conv_kernel)?,
                    delta: DeltaState::new(c.linear_value_heads, c.key_dim, c.value_dim)?,
                }
            });
        }
        Ok(layers)
    }
    pub(crate) fn new(c: &Config, context: usize, threads: usize) -> Result<Self> {
        let keys = c.linear_key_heads * c.key_dim;
        let values = c.linear_value_heads * c.value_dim;
        let channels = keys * 2 + values;
        let query_width = c.heads * c.head_dim;
        let layers = Self::states(c, context)?;
        let mut work = Workspace::new(
            c.hidden.max(c.intermediate).max(values).max(query_width),
            256 << 10,
            threads,
            1,
        )?;
        work.preserve_f32_operands();
        Ok(Self {
            layers,
            logits: memory::zeros(c.vocabulary)?,
            work,
            hidden: memory::zeros(c.hidden)?,
            normalized: memory::zeros(c.hidden)?,
            mixed: memory::zeros(channels.max(2 * query_width))?,
            q: memory::zeros(channels.max(query_width))?,
            k: memory::zeros(c.kv_heads * c.head_dim)?,
            v: memory::zeros(values.max(c.kv_heads * c.head_dim))?,
            z: memory::zeros(values.max(query_width))?,
            a: memory::zeros(c.linear_value_heads)?,
            b: memory::zeros(c.linear_value_heads)?,
            delta_q: memory::zeros(c.linear_value_heads * c.key_dim)?,
            delta_k: memory::zeros(c.linear_value_heads * c.key_dim)?,
            attention: memory::zeros(values.max(query_width))?,
            norm_head: memory::zeros(c.head_dim.max(c.value_dim))?,
            gate: memory::zeros(c.intermediate)?,
            up: memory::zeros(c.intermediate)?,
            projected: memory::zeros(c.hidden)?,
            final_norm: memory::zeros(c.hidden)?,
            position: 0,
            rotary: None,
            capacity: context,
        })
    }
}
