def get_overlapping_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Slices a long transcript into smaller chunks with a sliding window overlap.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        # Extract the slice of text
        chunk = text[start:end]
        
        # To make the vector contextually stronger, we clean up any leading/trailing whitespace
        chunks.append(chunk.strip())
        
        # Move the starting point forward, minus the overlap
        start += (chunk_size - overlap)
        
    return chunks