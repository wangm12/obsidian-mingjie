import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    The find_chunk_boundaries function works as follows:

    1. INITIALIZATION:
    - Calculates total file size in bytes
    - Divides file size by desired_num_chunks to get approximate chunk_size
    - Creates initial boundary guesses at uniform intervals

    2. BOUNDARY ADJUSTMENT:
    - For each internal boundary (not start/end):
        a. Seeks to the initial boundary position
        b. Reads mini-chunks of 4096 bytes
        c. Searches for the split_special_token in each mini-chunk
        d. When found, adjusts boundary to that exact position
        e. If EOF reached, sets boundary to file end

    3. FINAL PROCESSING:
    - Removes duplicate boundaries (sorted(set()))
    - Returns list of unique boundary positions

    KEY FEATURES:
    - Ensures chunks split at semantic boundaries (e.g., line breaks)
    - Prevents breaking tokens/words in the middle
    - Handles edge cases like small files or many requested chunks
    - Efficient: reads in 4KB chunks rather than entire file

    USE CASES:
    - Parallel text processing (each worker handles one chunk)
    - Tokenization of large corpora
    - Log file analysis
    - Distributed data processing
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def convert_word_to_bytes(word: str) -> tuple[bytes, ...]:
    """
    Convert a word to a tuple of individual bytes.
    Each element in the tuple is a single byte (bytes object of length 1).
    
    Example:
        convert_word_to_bytes("hi") -> (b'h', b'i')
        convert_word_to_bytes("€") -> (b'\xe2', b'\x82', b'\xac')  # 3 separate bytes
    """
    # First encode the entire word to UTF-8 bytes
    word_bytes = word.encode("utf-8")
    # Then split into individual single-byte objects
    return tuple(bytes([b]) for b in word_bytes)