use std::io::{self, Read};

mod private {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for f64 {}
}

pub(crate) trait FromRawBinary: private::Sealed {
    fn read_in<R: Read>(reader: R) -> io::Result<Self>
    where
        Self: Sized;

    fn read_size() -> usize;
}

impl FromRawBinary for u32 {
    #[inline]
    fn read_in<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut buf = [0u8; std::mem::size_of::<Self>()];
        reader.read_exact(&mut buf)?;
        Ok(Self::from_le_bytes(buf))
    }

    #[inline(always)]
    fn read_size() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl FromRawBinary for f64 {
    #[inline]
    fn read_in<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut buf = [0u8; std::mem::size_of::<Self>()];
        reader.read_exact(&mut buf)?;
        Ok(Self::from_le_bytes(buf))
    }

    #[inline(always)]
    fn read_size() -> usize {
        std::mem::size_of::<Self>()
    }
}

pub(crate) fn read_fort_record<R: Read, T: FromRawBinary>(
    mut reader: R,
    buffer: &mut [T],
) -> io::Result<()> {
    let expected_size = buffer.len() * <T as FromRawBinary>::read_size();
    let pre_size: u32 = FromRawBinary::read_in(&mut reader)?;
    if pre_size as usize != expected_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("requested {expected_size} bytes of data but next record has {pre_size}"),
        ));
    }
    for elt in buffer.iter_mut() {
        *elt = FromRawBinary::read_in(&mut reader)?;
    }
    let post_size: u32 = FromRawBinary::read_in(&mut reader)?;
    if post_size != pre_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected end of {pre_size} bytes record, found {post_size}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::read_fort_record;

    #[test]
    fn read_3_u32() {
        let raw_record = [
            12_u8, 0, 0, 0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 42, 0, 0, 0, 12, 0, 0,
            0,
        ];
        let mut buf = [0_u32; 3];
        read_fort_record(raw_record.as_slice(), &mut buf).expect("record well formed");
        assert_eq!(buf, [0x78563412, 0xf0debc9a, 42]);
    }

    #[test]
    fn read_2_f64() {
        let mut raw_record = [
            16_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xf0, 0x3f, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0,
        ];
        for (i, b) in std::f64::consts::PI.to_le_bytes().into_iter().enumerate() {
            raw_record[12 + i] = b;
        }
        let mut buf = [0_f64; 2];
        read_fort_record(raw_record.as_slice(), &mut buf).expect("record well formed");
        assert_eq!(buf, [1.0, std::f64::consts::PI]);
    }
}
