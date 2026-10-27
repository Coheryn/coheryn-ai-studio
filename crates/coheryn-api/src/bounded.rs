//! Count limits are enforced while parsing, before a hostile array is materialized.
use serde::{Deserialize, Deserializer, de};
use std::{fmt, marker::PhantomData};

pub(crate) fn sequence<'de, D, T, const LIMIT: usize>(d: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Visitor<T, const N: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const N: usize> de::Visitor<'de> for Visitor<T, N> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "an array of at most {N} items")
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut input: A) -> Result<Vec<T>, A::Error> {
            let mut items = Vec::new();
            while let Some(item) = input.next_element()? {
                if items.len() == N {
                    return Err(de::Error::custom("array exceeds the supported item limit"));
                }
                items.push(item);
            }
            Ok(items)
        }
    }
    d.deserialize_seq(Visitor::<T, LIMIT>(PhantomData))
}

pub(crate) fn optional_sequence<'de, D, T, const LIMIT: usize>(
    d: D,
) -> Result<Option<Vec<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Items<T, const N: usize>(Vec<T>);
    impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Items<T, N> {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            sequence::<D, T, N>(d).map(Self)
        }
    }
    Option::<Items<T, LIMIT>>::deserialize(d).map(|items| items.map(|items| items.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hostile_sequence_is_stopped_without_consuming_the_rest() {
        let mut read = 0;
        let stream = std::iter::from_fn(|| {
            read += 1;
            Some(String::new())
        });
        let input = de::value::SeqDeserializer::<_, de::value::Error>::new(stream);
        assert!(sequence::<_, String, 256>(input).is_err());
        assert_eq!(read, 257);
    }
}
