//! Image feature rows and decoder positions are distinct for spatial rotary models.
use super::image::{ImagePlan, Processor};
use crate::{Error, Result, memory};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ImagePositions {
    start: u32,
    next: u32,
    physical_tokens: usize,
    rows: u32,
    columns: u32,
    spatial_rotary: bool,
}
impl ImagePositions {
    /// Construct only the positional span. This does not qualify a model or admit KV memory.
    pub fn new(plan: &ImagePlan, start: u32) -> Result<Self> {
        let spatial = plan.profile == Processor::Qwen2VlStill;
        if plan.grid[0] != 1
            || plan.width == 0
            || plan.height == 0
            || plan.width > 32768
            || plan.height > 32768
            || plan.image_feature_tokens == 0
            || plan.image_feature_tokens > 16384
            || !plan.width.is_multiple_of(14)
            || !plan.height.is_multiple_of(14)
            || plan.grid[1] != (plan.height / 14) as usize
            || plan.grid[2] != (plan.width / 14) as usize
            || plan.patch_rows != memory::product(&plan.grid)?
            || (spatial && (!plan.width.is_multiple_of(28) || !plan.height.is_multiple_of(28)))
            || (!spatial && (plan.width != 336 || plan.height != 336))
        {
            return Err(Error::request(
                "image plan has inconsistent positional geometry",
            ));
        }
        let divisor = if spatial { 2 } else { 1 };
        let rows = plan.height / 14 / divisor;
        let columns = plan.width / 14 / divisor;
        let physical_tokens = memory::product(&[rows as usize, columns as usize])?;
        if physical_tokens != plan.image_feature_tokens {
            return Err(Error::request(
                "image feature count disagrees with positional grid",
            ));
        }
        let count = if spatial {
            rows.max(columns)
        } else {
            u32::try_from(physical_tokens)
                .map_err(|_| Error::resource("image position count overflow"))?
        };
        let next = start
            .checked_add(count)
            .filter(|n| *n <= i32::MAX as u32)
            .ok_or_else(|| Error::resource("image positions exceed supported decoder range"))?;
        Ok(Self {
            start,
            next,
            physical_tokens,
            rows,
            columns,
            spatial_rotary: spatial,
        })
    }
    pub fn physical_tokens(&self) -> usize {
        self.physical_tokens
    }
    pub fn position_count(&self) -> u32 {
        self.next - self.start
    }
    pub fn next_position(&self) -> u32 {
        self.next
    }
    fn position(&self, index: usize) -> [u32; 4] {
        if self.spatial_rotary {
            [
                self.start,
                self.start + index as u32 / self.columns,
                self.start + index as u32 % self.columns,
                0,
            ]
        } else {
            [self.start + index as u32; 4]
        }
    }
    /// Axes are in decoder order [temporal, vertical, horizontal, reserved].
    pub fn at(&self, index: usize) -> Result<[u32; 4]> {
        if index >= self.physical_tokens {
            return Err(Error::request("image position index outside span"));
        }
        Ok(self.position(index))
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = [u32; 4]> + '_ {
        (0..self.physical_tokens).map(|i| self.position(i))
    }
}
