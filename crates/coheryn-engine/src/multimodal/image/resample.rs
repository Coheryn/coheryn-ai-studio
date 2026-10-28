//! Coheryn separable Keys-cubic filtering, with versioned 22-bit coefficients.
//! Only the vertical support window is retained; there is no full horizontal image.
use super::*;
const ONE: i64 = 1 << 22;
#[derive(Clone, Default)]
struct Span {
    start: usize,
    count: usize,
    offset: usize,
}
struct Axis {
    spans: Buffer<Span>,
    weights: Buffer<i32>,
}
fn weight_bound(input: usize, output: usize) -> Result<usize> {
    input
        .checked_mul(4)
        .and_then(|n| output.checked_mul(6).and_then(|m| n.checked_add(m)))
        .ok_or_else(|| Error::resource("image filter coefficient overflow"))
}
pub(super) fn axis_bytes(input: usize, output: usize) -> Result<u64> {
    add(
        bytes(output, std::mem::size_of::<Span>())?,
        bytes(weight_bound(input, output)?, 4)?,
    )
}
pub(super) fn maximum_taps(input: usize, output: usize) -> Result<usize> {
    if input == 0 || output == 0 {
        return Err(Error::request("empty resize axis"));
    }
    Ok(input
        .min((4.0 * (input as f64 / output as f64).max(1.)).ceil() as usize + 2)
        .max(1))
}
fn cubic(distance: f64) -> f64 {
    let x = distance.abs();
    if x < 1. {
        ((1.5 * x - 2.5) * x) * x + 1.
    } else if x < 2. {
        (((x - 5.) * x + 8.) * x - 4.) * (-0.5)
    } else {
        0.
    }
}
impl Axis {
    fn new(input: usize, output: usize, budget: &Budget, cancel: &Cancellation) -> Result<Self> {
        let mut spans = Buffer::filled(output, Span::default(), budget)?;
        let mut weights = Buffer::filled(weight_bound(input, output)?, 0i32, budget)?;
        let scale = input as f64 / output as f64;
        let filter = scale.max(1.);
        let support = 2. * filter;
        let mut cursor = 0;
        for (position, span) in spans.iter_mut().enumerate() {
            if position % 64 == 0 {
                cancel.check()?;
            }
            let center = (position as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            if end <= start {
                return Err(Error::new(
                    ErrorCode::Internal,
                    "empty image filter support",
                ));
            }
            let sum: f64 = (start..end)
                .map(|i| cubic((i as f64 - center + 0.5) / filter))
                .sum();
            if !sum.is_finite() || sum <= 0. {
                return Err(Error::new(ErrorCode::Internal, "invalid image filter mass"));
            }
            if cursor + end - start > weights.len() {
                return Err(Error::resource(
                    "image filter support exceeded admitted bound",
                ));
            }
            *span = Span {
                start,
                count: end - start,
                offset: cursor,
            };
            for source in start..end {
                let coefficient = cubic((source as f64 - center + 0.5) / filter) / sum * ONE as f64;
                weights[cursor] = (coefficient + if coefficient < 0. { -0.5 } else { 0.5 }) as i32;
                cursor += 1;
            }
        }
        Ok(Self { spans, weights })
    }
}
fn rounded(value: i64) -> u8 {
    ((value + ONE / 2) >> 22).clamp(0, 255) as u8
}
pub(super) fn run(
    source: Source<'_>,
    plan: &ImagePlan,
    budget: &Budget,
    cancel: &Cancellation,
) -> Result<Buffer<u8>> {
    let (sw, sh) = source.info.oriented_dimensions();
    let sw = sw as usize;
    let sh = sh as usize;
    let rw = plan.resized_width as usize;
    let rh = plan.resized_height as usize;
    let width = plan.width as usize;
    let height = plan.height as usize;
    let mut output = Buffer::filled(memory::product(&[width, height, 3])?, 0u8, budget)?;
    if plan.profile == Processor::Llava15Pad336 {
        for pixel in output.as_chunks_mut::<3>().0 {
            pixel.copy_from_slice(&[122, 116, 104]);
        }
    }
    let ox = (width - rw) / 2;
    let oy = (height - rh) / 2;
    if sw == rw && sh == rh {
        for y in 0..rh {
            cancel.check()?;
            for x in 0..rw {
                for c in 0..3 {
                    output[((y + oy) * width + x + ox) * 3 + c] = source.pixel(x, y, c);
                }
            }
        }
        return Ok(output);
    }
    let x_axis = Axis::new(sw, rw, budget, cancel)?;
    let y_axis = Axis::new(sh, rh, budget, cancel)?;
    let taps = maximum_taps(sh, rh)?;
    let row_bytes = memory::product(&[rw, 3])?;
    let mut ring = Buffer::filled(memory::product(&[taps, row_bytes])?, 0u8, budget)?;
    let mut markers = Buffer::filled(taps, usize::MAX, budget)?;
    for (y, span) in y_axis.spans.iter().enumerate() {
        cancel.check()?;
        for sy in span.start..span.start + span.count {
            let slot = sy % taps;
            if markers[slot] == sy {
                continue;
            }
            cancel.check()?;
            let row = &mut ring[slot * row_bytes..(slot + 1) * row_bytes];
            for (x, filter) in x_axis.spans.iter().enumerate() {
                if x % 64 == 0 {
                    cancel.check()?;
                }
                for channel in 0..3 {
                    let mut sum = 0i64;
                    for i in 0..filter.count {
                        sum += source.pixel(filter.start + i, sy, channel) as i64
                            * x_axis.weights[filter.offset + i] as i64;
                    }
                    row[x * 3 + channel] = rounded(sum);
                }
            }
            markers[slot] = sy;
        }
        for x in 0..rw {
            if x % 64 == 0 {
                cancel.check()?;
            }
            for channel in 0..3 {
                let mut sum = 0i64;
                for i in 0..span.count {
                    sum += ring[(span.start + i) % taps * row_bytes + x * 3 + channel] as i64
                        * y_axis.weights[span.offset + i] as i64;
                }
                output[((y + oy) * width + x + ox) * 3 + channel] = rounded(sum);
            }
        }
    }
    Ok(output)
}
