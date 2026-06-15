def align_offset(offset: int, alignment: int = 16) -> int:
    """16-byte alignment for memory safety"""
    return (offset + alignment - 1) & ~(alignment - 1)