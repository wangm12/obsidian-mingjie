import regex as re
import collections
import multiprocessing
import pickle
import os
from cs336_basics.tokenizer.utils import find_chunk_boundaries, convert_word_to_bytes

def initialize_vocab(vocab: dict, initial_vocab_size: int):
    """
    Initialize vocabulary with byte values from 0 to initial_vocab_size-1.
    
    This function creates the base vocabulary consisting of individual byte values,
    which serves as the foundation for BPE tokenization. Each byte value (0-255)
    gets its own token ID.
    
    Args:
        vocab (dict): Dictionary mapping token IDs to byte sequences (modified in place)
        initial_vocab_size (int): Number of initial byte tokens (typically 256)
    
    Returns:
        None (modifies vocab in place)
    
    Example:
        >>> vocab = {}
        >>> initialize_vocab(vocab, 256)
        >>> vocab[65]  # Returns b'A' (ASCII 65)
    """
    # vocab: index -> byte
    for i in range(initial_vocab_size):
        b = bytes([i])
        vocab[i] = b
    return

def add_special_tokens(vocab, special_tokens: list[str]):
    """
    Add special tokens to the vocabulary if they don't already exist.
    
    Special tokens are added as complete units and won't be split during tokenization.
    Common special tokens include <|endoftext|>, <pad>, <unk>, etc.
    
    Args:
        vocab (dict): Dictionary mapping token IDs to byte sequences (modified in place)
        special_tokens (list[str]): List of special token strings to add
    
    Returns:
        None (modifies vocab in place)
    
    Example:
        >>> vocab = {}
        >>> initialize_vocab(vocab, 256)
        >>> add_special_tokens(vocab, ["<|endoftext|>", "<pad>"])
        >>> vocab[256]  # Returns b'<|endoftext|>'
    """
    for special_token in special_tokens:
        special_token_byte = special_token.encode("utf-8")
        # Check if special token already exists in vocab values
        if special_token_byte not in vocab.values():
            i = len(vocab)
            vocab[i] = special_token_byte
    return 

def initialize_vocab_and_special_tokens(initial_vocab_size: int, special_tokens: list[str]):
    """
    Create and initialize vocabulary with both byte tokens and special tokens.
    
    This is a convenience function that combines vocabulary initialization with
    special token addition in a single step.
    
    Args:
        initial_vocab_size (int): Number of initial byte tokens (typically 256)
        special_tokens (list[str]): List of special token strings to add
    
    Returns:
        dict: vocab - Maps token IDs to byte sequences
    
    Example:
        >>> vocab = initialize_vocab_and_special_tokens(256, ["<|endoftext|>"])
        >>> len(vocab)  # Returns 257 (256 bytes + 1 special token)
    """
    vocab = {}
    initialize_vocab(vocab, initial_vocab_size)
    add_special_tokens(vocab, special_tokens)
    return vocab

########################################################

def build_word_counter(input_path: str, chunk_boundaries: list[int], special_tokens: list[str]):
    """
    Build a counter of pre-tokenized words from the input file using multiprocessing.
    
    This function splits the input text into words using a regex pattern (GPT-2 style),
    converts them to byte sequences, and counts their frequencies. It uses multiple
    processes to handle large files efficiently.
    
    Args:
        input_path (str): Path to the input text file
        chunk_boundaries (list[int]): Byte positions defining chunk boundaries for parallel processing
        special_tokens (list[str]): Special tokens to preserve during tokenization
    
    Returns:
        collections.Counter: Counter mapping word byte tuples to their frequencies
    
    Example:
        >>> boundaries = [0, 1000000, 2000000]  # Process file in 2 chunks
        >>> counter = build_word_counter("data.txt", boundaries, ["<|endoftext|>"])
        >>> counter[(b'H', b'e', b'l', b'l', b'o')]  # Returns frequency of "Hello"
    """
    # REGEX PATTERN - HARDCODED FOR NOW
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens_regex = "|".join(re.escape(token) for token in special_tokens)
    
    # use counter bcs its easy to update and combine
    word_counter = collections.Counter()
    
    # need to use multiprocessing.Manager().Queue() bcs this queue is shared between processes; and need an individual process to manage
    queue = multiprocessing.Manager().Queue(maxsize=len(chunk_boundaries) - 1)
    processes = []
    
    for i in range(len(chunk_boundaries) - 1):
        start, end = chunk_boundaries[i], chunk_boundaries[i+1]
        p = multiprocessing.Process(
            target=process_chunk, 
            args=(queue, input_path, word_counter, start, end, PAT, special_tokens_regex)
        )
        processes.append(p)
        p.start()
                
    for p in processes:
        p.join()
    
    for _ in range(len(processes)):
        try:
            word_counter += queue.get(timeout=5)
        except multiprocessing.TimeoutError:
            print("Multiprocessing Timeout error")
    
    return word_counter

def process_chunk(queue: multiprocessing.Queue, input_path: str, word_counter: collections.Counter, chunk_boundary_start: int, chunk_boundary_end: int, word_split_pattern: str, special_tokens_regex: str):
    """
    Process a single chunk of the input file in a separate process.
    
    This function is called by build_word_counter to process chunks in parallel.
    It reads a specific byte range from the file, splits it into words, and counts
    their frequencies.
    
    Args:
        queue (multiprocessing.Queue): Queue to put the resulting word counter
        input_path (str): Path to the input text file
        word_counter (collections.Counter): Counter to store word frequencies (unused template)
        chunk_boundary_start (int): Starting byte position of the chunk
        chunk_boundary_end (int): Ending byte position of the chunk
        word_split_pattern (str): Regex pattern for splitting text into words
        special_tokens_regex (str): Regex pattern for special tokens
    
    Returns:
        None (puts result in queue)
    
    Example:
        >>> # This is typically called internally by build_word_counter
        >>> queue = multiprocessing.Manager().Queue()
        >>> process_chunk(queue, "data.txt", Counter(), 0, 1000, PAT, "<|endoftext|>")
        >>> result = queue.get()  # Get the word counter for this chunk
    """
    with open(input_path, "rb") as f:
        f.seek(chunk_boundary_start)
        chunk = f.read(chunk_boundary_end - chunk_boundary_start)
        
    # split the chunk by pattern; need to decode first because we are using regex
    split_chunks = re.split(special_tokens_regex, chunk.decode("utf-8"))
    
    # update the word counter
    for chunk in split_chunks:
        for match in re.finditer(word_split_pattern, chunk):
            # convert the whole match to a string
            word = match.group()
            # convert the word to a tuple of bytes
            word_bytes = convert_word_to_bytes(word)
            # update the word counter
            word_counter[word_bytes] += 1
    
    # put the word counter into the queue
    queue.put(word_counter)
    return
                
def pre_tokenize(input_path: str, chunk_size: int, special_tokens: list[str], split_special_token: bytes):
    """
    Perform pre-tokenization on the input file to create word-level tokens.
    
    This function divides the input file into chunks for parallel processing and
    counts the frequency of each pre-tokenized word. Pre-tokenization is the first
    step before applying BPE merges.
    
    Args:
        input_path (str): Path to the input text file
        chunk_size (int): Number of chunks to divide the file into
        special_tokens (list[str]): Special tokens to preserve during tokenization
        split_special_token (bytes): Special token used to find chunk boundaries
    
    Returns:
        collections.Counter: Counter mapping word byte tuples to their frequencies
    
    Example:
        >>> counter = pre_tokenize("corpus.txt", 10, ["<|endoftext|>"], b"<|endoftext|>")
        >>> # Returns word frequencies from the corpus split into 10 chunks
    """
    chunk_boundaries = []
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, chunk_size, split_special_token)
    
    word_counter = build_word_counter(input_path, chunk_boundaries, special_tokens)
    return word_counter

########################################################

def build_pair_counter(word_counter: collections.Counter):
    """
    Build a counter of byte pairs and an index mapping pairs to words containing them.
    
    This function analyzes all words to find adjacent byte pairs and their frequencies.
    It also maintains an index of which words contain each pair, which is crucial
    for efficient BPE merge operations.
    
    Args:
        word_counter (collections.Counter): Counter mapping word byte tuples to frequencies
    
    Returns:
        tuple: (pair_counter, pair_to_words) where:
            - pair_counter (dict): Maps byte pairs to their total frequencies
            - pair_to_words (dict): Maps byte pairs to sets of words containing them
    
    Example:
        >>> word_counter = Counter({(b'h', b'e', b'l', b'l', b'o'): 5})
        >>> pair_counter, pair_to_words = build_pair_counter(word_counter)
        >>> pair_counter[(b'l', b'l')]  # Returns 5 (frequency of 'll' pair)
        >>> pair_to_words[(b'l', b'l')]  # Returns {(b'h', b'e', b'l', b'l', b'o')}
    """
    # key: tuple of bytes, value: frequency
    pair_counter = collections.defaultdict(int)
    # key: tuple of bytes, value: set of word_bytes containing that pair
    pair_to_words = collections.defaultdict(set)
    
    for word_bytes, count in word_counter.items():
        for i in range(len(word_bytes) - 1):
            pair = tuple(word_bytes[i:i+2])
            pair_counter[pair] += count
            pair_to_words[pair].add(word_bytes)
    
    return pair_counter, pair_to_words

def merge_pair(
    most_frequent_pair: tuple[bytes, bytes],
    pair_bytes: bytes,
    word_counter: collections.Counter,
    pair_counter: collections.defaultdict,
    pair_to_words: collections.defaultdict,
):
    """
    Merge the most frequent byte pair in all words containing it.
    
    This function performs a single BPE merge operation. It finds all words containing
    the specified pair, merges the pair into a single token, and updates all relevant
    data structures (word counter, pair counter, and pair-to-words index).
    
    Args:
        most_frequent_pair (tuple[bytes, bytes]): The byte pair to merge
        pair_bytes (bytes): The merged representation of the pair
        word_counter (collections.Counter): Word frequency counter (modified in place)
        pair_counter (defaultdict): Pair frequency counter (modified in place)
        pair_to_words (defaultdict): Pair-to-words index (modified in place)
    
    Returns:
        None (modifies data structures in place)
    
    Example:
        >>> # If most_frequent_pair is (b'l', b'l') and appears in "hello"
        >>> # After merging: (b'h', b'e', b'l', b'l', b'o') becomes (b'h', b'e', b'll', b'o')
    """
    # get the words_bytes that contain the pair; need to copy bcs we are modifying the set
    affected_words = pair_to_words[most_frequent_pair].copy()
    
    # merge the pair
    for word_bytes in affected_words:
        # edge case check; as we are modifying the set, we need to check if the word_bytes is still in the word_counter
        if word_bytes not in word_counter:
            continue
        
        count = word_counter[word_bytes]
        
        # find the matching pair
        matches = []
        for i in range(len(word_bytes) - 1):
            if word_bytes[i:i+2] == most_frequent_pair:
                matches.append(i)
        
        # if there is no matching pair, continue the loop
        if not matches:
            continue
        
        # build the new word_bytes
        new_word_bytes = []
        i = 0
        while i < len(word_bytes):
            if i in matches:
                new_word_bytes.append(pair_bytes)
                i += 2
            else:
                new_word_bytes.append(word_bytes[i])
                i += 1
        new_word_bytes = tuple(new_word_bytes)
        
        # update the pair counter; remove old pairs
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i+1])  
            pair_counter[pair] -= count
            pair_to_words[pair].discard(word_bytes)
        
        # update the pair counter; add new pairs
        for i in range(len(new_word_bytes) - 1):
            pair = (new_word_bytes[i], new_word_bytes[i+1])  # Fix: use new_word_bytes, not word_bytes
            pair_counter[pair] += count
            pair_to_words[pair].add(new_word_bytes)
        
        # update the word counter; delete old word_bytes and add new word_bytes
        del word_counter[word_bytes]
        word_counter[new_word_bytes] += count
    return

def merge_pairs(vocab: dict, word_counter: collections.Counter, vocab_size: int):
    """
    Perform BPE merges until reaching the target vocabulary size.
    
    This is the core BPE algorithm. It repeatedly finds the most frequent byte pair,
    merges it into a single token, and updates the vocabulary. The process continues
    until the vocabulary reaches the desired size.
    
    Args:
        vocab (dict): Token ID to bytes mapping (modified in place)
        word_counter (collections.Counter): Word frequency counter (modified in place)
        vocab_size (int): Target vocabulary size
    
    Returns:
        list: List of merged byte sequences in order of merging
    
    Example:
        >>> vocab = initialize_vocab_and_special_tokens(256, [])
        >>> word_counter = Counter({(b'h', b'e', b'l', b'l', b'o'): 10})
        >>> merges = merge_pairs(vocab, word_counter, 260)
        >>> # Returns list of 4 merges, e.g., [b'll', b'he', b'llo', b'hello']
    """
    merges = []
    merge_counts = vocab_size - len(vocab)
    
    pair_counter, pair_to_words = build_pair_counter(word_counter)
    
    while merge_counts > 0:
        # find the highest frequency pair
        most_frequent_pair_count = max(pair_counter.values())
        most_frequent_pair = max([pair for pair, freq in pair_counter.items() if freq == most_frequent_pair_count])

        pair_bytes = most_frequent_pair[0] + most_frequent_pair[1]
        pair_bytes_index = len(vocab)
        
        # add the pair to the vocab and merges
        vocab[pair_bytes_index] = pair_bytes
        merges.append(most_frequent_pair)
        
        # merge the pair
        merge_pair(most_frequent_pair, pair_bytes, word_counter, pair_counter, pair_to_words)
        
        merge_counts -= 1
    
    return merges
    
########################################################

def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the given corpus.
    
    This is the main entry point for training a BPE tokenizer. It performs three steps:
    1. Initialize vocabulary with byte tokens and special tokens
    2. Pre-tokenize the corpus into words and count frequencies
    3. Apply BPE merges to build the final vocabulary
    
    The function automatically determines optimal chunk sizes based on file size
    for efficient parallel processing of large corpora.
    
    Args:
        input_path (str): Path to the training corpus text file
        vocab_size (int): Target vocabulary size (must be >= 256 + len(special_tokens))
        special_tokens (list[str]): List of special tokens to include in vocabulary
    
    Returns:
        tuple: (vocab, merges) where:
            - vocab (dict): Final vocabulary mapping token IDs to byte sequences
            - merges (list): List of byte sequences representing merge operations
    
    Example:
        >>> vocab, merges = train_bpe_tokenizer(
        ...     "corpus.txt",
        ...     vocab_size=10000,
        ...     special_tokens=["<|endoftext|>", "<pad>"]
        ... )
        >>> len(vocab)  # Returns 10000
        >>> len(merges)  # Returns 9742 (10000 - 256 - 2 special tokens)
    """
    # STEP 1: initialize
    vocab = initialize_vocab_and_special_tokens(256, special_tokens)
    
    # STEP 2: pre-tokenization
    file_size = os.path.getsize(input_path)
    # Use fewer chunks for smaller files, more for larger files
    # Aim for chunks of ~10-50MB each
    optimal_chunk_mb = 20  # MB per chunk
    chunk_size = max(4, min(100, file_size // (optimal_chunk_mb * 1024 * 1024)))
    print(f"Using {chunk_size} chunks for file of size {file_size / (1024*1024):.2f} MB")
    
    split_special_token = b"<|endoftext|>"
    word_counter = pre_tokenize(input_path, chunk_size, special_tokens, split_special_token)
    
    # STEP 3: BPE merge
    merges = merge_pairs(vocab, word_counter, vocab_size)
    
    return vocab, merges

########################################################

if __name__ == "__main__": 
    from time import time
    
    # Define test configurations
    dataset_configs = [
        {
            "name": "TinyStoriesV2-GPT4-train",
            "path": "../../data/TinyStoriesV2-GPT4-train.txt",
            "output_path": "../../data/output",
            "vocab_size": 10000,
            "special_tokens": ["<|endoftext|>"]
        },
        {
            "name": "TinyStoriesV2-GPT4-valid",
            "path": "../../data/TinyStoriesV2-GPT4-valid.txt",
            "output_path": "../../data/output",
            "vocab_size": 10000,
            "special_tokens": ["<|endoftext|>"]
        },
        {
            "name": "owt_train",
            "path": "../../data/owt_train.txt",
            "output_path": "../../data/output",
            "vocab_size": 32000,
            "special_tokens": ["<|endoftext|>"]
        },
        {
            "name": "owt_valid",
            "path": "../../data/owt_valid.txt",
            "output_path": "../../data/output",
            "vocab_size": 32000,
            "special_tokens": ["<|endoftext|>"]
        }
    ]
    
    # Store results for table display
    results = []
    
    # Test each dataset
    for config in dataset_configs:
        print(f"\nProcessing {config['name']}...")
        start_time = time()
        vocab, merges = train_bpe_tokenizer(
            config['path'], 
            config['vocab_size'], 
            config['special_tokens']
        )
        end_time = time()
        time_consumed = end_time - start_time
        
        results.append({
            "dataset": config['name'],
            "time": time_consumed,
            "merges": len(merges)
        })
        
        # write the vocab and merges to files
        with open(os.path.join(config['output_path'], f"{config['name']}_vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)
        with open(os.path.join(config['output_path'], f"{config['name']}_merges.pkl"), "wb") as f:
            pickle.dump(merges, f)
        
        print(f"Completed {config['name']} in {time_consumed:.2f} seconds")
    
    # Print results table
    """
    RESULTS TABLE
    ======================================================================
    Dataset                        | Time (seconds)  | Merges    
    ----------------------------------------------------------------------
    TinyStoriesV2-GPT4-train       | 38.58           | 9743      
    TinyStoriesV2-GPT4-valid       | 2.70            | 9743      
    owt_train                      | 1625.86         | 31743     
    owt_valid                      | 112.64          | 31743     
    ======================================================================
    """
    print("\nRESULTS TABLE")
    print("="*70)
    print(f"{'Dataset':<30} | {'Time (seconds)':<15} | {'Merges':<10}")
    print("-"*70)
    for result in results:
        print(f"{result['dataset']:<30} | {result['time']:<15.2f} | {result['merges']:<10}")
    print("="*70)