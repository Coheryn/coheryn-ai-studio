# Certified CPU dot reduction

The candidate dot path keeps F32 products and the original F32 result. Four independent F64 sums let the compiler schedule parallel arithmetic using safe Rust. It allocates no heap memory and uses no explicit ISA intrinsics, unsafe code, fused products or fast-math flags. The initial measured dispatch interval is 384–16,384 terms; other lengths retain the original reduction.

## Rounding certificate

Let `pᵢ` be each already-rounded F32 product, `S = Σpᵢ`, `A = Σ|pᵢ|`, `u = 2⁻⁵³` and `γₙ = nu/(1−nu)`. Both summation trees have absolute error at most `γₙ A`; the positive magnitude reduction `Â` satisfies `Â ≥ (1−γₙ) A`. These use the general summation bounds in [Higham, 1993, §2–3](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf).

Therefore the reference and candidate F64 sums differ by at most `2γₙ Â/(1−γₙ)`. For `256 ≤ n ≤ 2²⁰`, the implementation's `16 n EPSILON Â` conservatively exceeds that bound, including its evaluation roundoff. Finite F32 products widen exactly; their F64 sums cannot underflow or overflow in this domain. Nonfinite products reject the certificate.

The dispatch interval is a performance choice inside this numerical domain. Initial M1 calibration found an unstable crossover at 256 terms and regressions at 32,768/65,536 terms; those lengths retain the original reduction. Cross-platform measurements remain required, and cancellation-heavy inputs can still incur reference-fallback overhead within the selected interval.

The interval endpoints are rounded outward with `next_down`/`next_up`. Acceptance requires identical F32 endpoint bits. Monotone [Rust F64-to-F32 rounding](https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast) then proves the reference produces those same bits. Ambiguous ties, cancellation, signed-zero boundaries and nonfinite values fall back to the unchanged reference expression. This certifies reference compatibility rather than a different real-number summation result. Existing independent fixture tolerances remain unchanged.

## Evidence and measurements

Ordinary tests compare reference bits across vector boundaries/tails, broad finite exponents, unequal lengths, cancellation, halfway rounding, subnormals, signed zero and nonfinite products. Independent model fixtures remain required.

The bounded `dot_bench` engine example compares both paths on the same warm synthetic rows, alternates measurement order and reports all seven samples for fourteen normal widths plus a cancellation workload. Widths include the hidden/FFN dimensions of the pinned small Qwen models, small-width crossover cases and wider reductions. The cancellation case exposes fallback overhead. These are kernel observations; shared-runner timing is not controlled model-throughput evidence. Three-platform parity, observed performance and model checks remain required before accepting this candidate.

Run the guarded build and measurement with a fresh output path:

```sh
python3 tools/check_cpu_dot.py --report target/cpu-dot-observations.json
```

The controller uses the existing host limits, one offline Cargo job, owned process groups, deadlines and bounded output. It retains compiler, native executable, source, log and observer identities. Incomplete cases/samples, failed commands, source changes or unhealthy observations fail explicitly. CI runs this after ordinary native checks on each supported CPU platform. Full independent model comparisons and the text-adapter benchmark matrix remain separate requirements.

The reviewed [M1 checkpoint](../fixtures/qualification/cpu-certified-dot-m1-v1.json) and [Linux/Windows checkpoint](../fixtures/qualification/cpu-certified-dot-github-v1.json) retain original reports for revision `360bac1`. All fifteen synthetic cases preserve reference bits on each target. The local M1 run passed 68 release library tests and strict all-target Clippy; Linux and Windows passed 264 and 260 ordinary workspace tests respectively, each with ten explicit model opt-ins ignored. The macOS GitHub job stopped at startup load admission. The earlier ordinary CI checker did not continuously guard its full suite; the later checker adds that guard, release profiles and a serial test harness.

Eligible normal-width rows showed lower median kernel time in these runs, while the cancellation-heavy fallback was slower. These observations support the bounded dispatch choice; they do not establish model speedups. Exact source identities, unavailable measurements and remaining full-model/platform/release checks stay explicit in the records. The original M1 snapshots also retain two Finder metadata hashes; their exclusion from authored-source comparison is disclosed rather than editing the reports.
