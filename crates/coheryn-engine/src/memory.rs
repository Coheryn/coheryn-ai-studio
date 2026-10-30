use crate::{Error, ModelConfig, Result};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory: u64,
    pub context: usize,
}
impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 512 << 20,
            context: 4096,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub reserved: u64,
    pub peak_reserved: u64,
    pub limit: u64,
    /// Observed capacities of engine session buffers; excludes allocator overhead and libraries.
    #[serde(default)]
    pub session_buffer_capacity_bytes: u64,
    #[serde(default)]
    pub peak_session_buffer_capacity_bytes: u64,
}

#[derive(Clone)]
pub struct Budget(Arc<Mutex<MemoryStats>>);
pub struct Reservation {
    budget: Budget,
    bytes: u64,
}
impl Reservation {
    pub(crate) fn absorb(&mut self, mut other: Self) -> Result<()> {
        if !Arc::ptr_eq(&self.budget.0, &other.budget.0) {
            return Err(Error::resource(
                "cannot transfer reservations across budgets",
            ));
        }
        self.bytes = self
            .bytes
            .checked_add(other.bytes)
            .ok_or_else(|| Error::resource("reservation transfer overflow"))?;
        other.bytes = 0;
        Ok(())
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Atomically grow this reservation; failure leaves its held capacity unchanged.
    pub fn extend(&mut self, bytes: u64) -> Result<()> {
        let additional = self.budget.reserve(bytes)?;
        self.absorb(additional)
    }
    /// Transfer part of an existing atomic reservation without releasing/reacquiring bytes.
    pub fn split(&mut self, bytes: u64) -> Result<Self> {
        if bytes > self.bytes {
            return Err(Error::resource("reservation split exceeds held capacity"));
        }
        self.bytes -= bytes;
        Ok(Self {
            budget: self.budget.clone(),
            bytes,
        })
    }
    /// Check owner identity and capacity before transferring workspace admission.
    pub fn covers(&self, budget: &Budget, bytes: u64) -> bool {
        Arc::ptr_eq(&self.budget.0, &budget.0) && self.bytes >= bytes
    }
}
pub(crate) struct BufferAccounting {
    budget: Budget,
    bytes: u64,
    limit: u64,
}
impl BufferAccounting {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(crate) fn relimit(&mut self, limit: u64) -> Result<()> {
        if self.bytes > limit {
            return Err(Error::resource("buffer transfer exceeds reservation"));
        }
        self.limit = limit;
        Ok(())
    }
    pub(crate) fn split(&mut self, bytes: u64) -> Result<Self> {
        if bytes > self.bytes {
            return Err(Error::resource("buffer transfer exceeds observed capacity"));
        }
        self.bytes -= bytes;
        self.limit -= bytes;
        Ok(Self {
            budget: self.budget.clone(),
            bytes,
            limit: bytes,
        })
    }
    pub(crate) fn grow(&mut self, bytes: u64) -> Result<()> {
        let next = self
            .bytes
            .checked_add(bytes)
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| Error::resource("observed session buffers exceed reservation"))?;
        let mut stats = self
            .budget
            .0
            .lock()
            .map_err(|_| Error::resource("budget lock poisoned"))?;
        stats.session_buffer_capacity_bytes = stats
            .session_buffer_capacity_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::resource("buffer accounting overflow"))?;
        stats.peak_session_buffer_capacity_bytes = stats
            .peak_session_buffer_capacity_bytes
            .max(stats.session_buffer_capacity_bytes);
        self.bytes = next;
        Ok(())
    }
}
impl Drop for BufferAccounting {
    fn drop(&mut self) {
        if let Ok(mut stats) = self.budget.0.lock() {
            stats.session_buffer_capacity_bytes -= self.bytes;
        }
    }
}
impl Budget {
    pub fn new(limit: u64) -> Self {
        Self(Arc::new(Mutex::new(MemoryStats {
            limit,
            ..MemoryStats::default()
        })))
    }
    pub fn reserve(&self, bytes: u64) -> Result<Reservation> {
        let mut s = self
            .0
            .lock()
            .map_err(|_| Error::resource("budget lock poisoned"))?;
        let next = s
            .reserved
            .checked_add(bytes)
            .filter(|n| *n <= s.limit)
            .ok_or_else(|| {
                Error::resource(format!(
                    "need {bytes} additional bytes; {} of {} reserved",
                    s.reserved, s.limit
                ))
            })?;
        s.reserved = next;
        s.peak_reserved = s.peak_reserved.max(next);
        Ok(Reservation {
            budget: self.clone(),
            bytes,
        })
    }
    pub fn stats(&self) -> MemoryStats {
        *self.0.lock().expect("budget mutex")
    }
    pub(crate) fn track_buffers(&self, limit: u64) -> BufferAccounting {
        BufferAccounting {
            budget: self.clone(),
            bytes: 0,
            limit,
        }
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        if let Ok(mut s) = self.budget.0.lock() {
            s.reserved -= self.bytes;
        }
    }
}

pub fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, n| {
        acc.checked_mul(*n)
            .ok_or_else(|| Error::resource("allocation size overflow"))
    })
}
pub fn zeros(n: usize) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::resource("float buffer allocation failed"))?;
    v.resize(n, 0.0);
    Ok(v)
}
pub fn zeros_f64(n: usize) -> Result<Vec<f64>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::resource("attention accumulator allocation failed"))?;
    v.resize(n, 0.);
    Ok(v)
}
pub fn session_bytes(c: &ModelConfig, context: usize) -> Result<u64> {
    session_bytes_with_io(c, context, 256 << 10)
}
pub fn session_bytes_with_io(c: &ModelConfig, context: usize, io_bytes: usize) -> Result<u64> {
    session_bytes_for(c, context, io_bytes, 1)
}
pub fn worker_bytes(threads: usize) -> Result<usize> {
    if !(1..=16).contains(&threads) {
        return Err(Error::request("CPU threads must be in 1..=16"));
    }
    Ok(if threads == 1 {
        0
    } else {
        threads * ((512 + 64) << 10)
    })
}
pub fn session_bytes_for(
    c: &ModelConfig,
    context: usize,
    io_bytes: usize,
    threads: usize,
) -> Result<u64> {
    if !(4096..=256 << 10).contains(&io_bytes) || !io_bytes.is_power_of_two() {
        return Err(Error::request(
            "I/O tile must be a power of two in 4..=256 KiB",
        ));
    }
    let kv = product(&[2, c.layers, c.kv_heads, c.head_dim, context, 4])?;
    // Scratch, sampling vectors, attention accumulator, token bookkeeping, layer vectors.
    let work_elements = [
        (c.hidden, 16),
        (c.intermediate, 3),
        (c.vocabulary, 8),
        (context, 8),
    ]
    .into_iter()
    .try_fold(0usize, |acc, (n, m)| {
        n.checked_mul(m)
            .and_then(|v| acc.checked_add(v))
            .ok_or_else(|| Error::resource("workspace overflow"))
    })?;
    let work = product(&[work_elements, 4])?;
    let io = c
        .hidden
        .max(c.intermediate)
        .checked_mul(4)
        .and_then(|n| n.checked_add(3 * io_bytes))
        .ok_or_else(|| Error::resource("I/O workspace overflow"))?;
    let activation = c
        .hidden
        .max(c.intermediate)
        .div_ceil(256)
        .checked_mul(292)
        .ok_or_else(|| Error::resource("activation reservation overflow"))?;
    let workers = worker_bytes(threads)?;
    kv.checked_add(work)
        .and_then(|v| v.checked_add(io))
        .and_then(|v| v.checked_add(activation))
        .and_then(|v| v.checked_add(workers))
        .and_then(|v| v.checked_add(2 << 20))
        .map(|v| v as u64)
        .ok_or_else(|| Error::resource("session reservation overflow"))
}
pub fn model_bytes(d: &crate::Descriptor, c: &ModelConfig) -> Result<u64> {
    d.data_offset
        .checked_mul(16)
        .and_then(|v| v.checked_add((c.vocabulary as u64).checked_mul(1024)?))
        .ok_or_else(|| Error::resource("model metadata reservation overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservation_is_atomic_and_released() {
        let budget = Budget::new(100);
        let a = budget.reserve(80).unwrap();
        assert!(budget.reserve(21).is_err());
        drop(a);
        assert_eq!(budget.stats().reserved, 0);
        assert_eq!(budget.stats().peak_reserved, 80);
    }
    #[test]
    fn checked_product_rejects_overflow() {
        assert!(product(&[usize::MAX, 2]).is_err());
    }
}
