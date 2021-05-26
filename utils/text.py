def supersplit(string: str, delimiter: str):
    """Like str.split, but keeps delimiter and discards empty bits."""
    return [
        bit
        for split in string.split(delimiter)
        for bit in [delimiter, split]
        if len(bit) > 0
    ][1:]
