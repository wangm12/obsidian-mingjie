import regex as re
from typing import Iterable, Iterator
class BPE_Tokenizer:
    """Byte Pair Encoding tokenizer implementation.
    
    This tokenizer uses BPE merges to encode text into token IDs and can handle
    special tokens that should not be split during tokenization.
    
    Example:
        tokenizer = BPE_Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
        ids = tokenizer.encode("Hello world!")
        text = tokenizer.decode(ids)
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """Initialize the BPE tokenizer with vocabulary, merges, and optional special tokens.
        
        Args:
            vocab: Dictionary mapping token IDs to byte sequences
            merges: List of byte pairs to merge in order of priority
            special_tokens: Optional list of special tokens that should not be split
        """
        self.vocab = vocab
        self.vocab_inverse = {v:k for k, v in self.vocab.items()}
        self.merges = merges
        self.merges_inverse = {tuple(merge):i for i, merge in enumerate(merges)}
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to the vocabulary if they're not already present.
        
        Special tokens are added to the end of the vocabulary with sequential IDs.
        """
        if self.special_tokens is None:
            return
        
        for special_token in self.special_tokens:
            special_token_byte = special_token.encode("utf-8")
            # if special_token_byte not in vocab, add in
            if special_token_byte not in self.vocab_inverse:
                i = len(self.vocab)
                self.vocab[i] = special_token_byte
                self.vocab_inverse[special_token_byte] = i
        return 
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Create a tokenizer from saved vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to pickled vocabulary dictionary
            merges_filepath: Path to pickled merges list
            special_tokens: Optional list of special tokens
            
        Returns:
            BPE_Tokenizer instance
            
        Example:
            tokenizer = BPE_Tokenizer.from_files("vocab.pkl", "merges.pkl", ["<|endoftext|>"])
        """
        import pickle
        
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
            
        return cls(vocab, merges, special_tokens)
        
    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs, handling special tokens if configured.
        
        Args:
            text: The input text to tokenize
            
        Returns:
            List of token IDs representing the encoded text
        """
        text_token_ids = []
        start_idx = 0

        if self.special_tokens:
            special_tokens_regex = "|".join(re.escape(token) for token in self.special_tokens)

            for match in re.finditer(special_tokens_regex, text):
                # Process text before the special token
                chunk = text[start_idx:match.start()]
                start_idx = match.end()
                
                text_token_ids.extend(self._encode_chunk(chunk))
                    
                # Add the special token to the encoding list
                text_token_ids.append(self.vocab_inverse[match.group().encode("utf-8")])
            
            # Process any remaining text after the last special token
            if start_idx < len(text):
                final_chunk = text[start_idx:]
                text_token_ids.extend(self._encode_chunk(final_chunk))
        else:
            text_token_ids.extend(self._encode_chunk(text))
        return text_token_ids
    
    def _encode_chunk(self, chunk: str):
        """Encode a chunk of text without special tokens using GPT-2 regex pattern.
        
        Args:
            chunk: Text chunk to encode (without special tokens)
            
        Returns:
            List of token IDs for the chunk
            
        Note:
            Uses GPT-2's regex pattern to split text into words before encoding
        """
        chunk_token_ids = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for word_match in re.finditer(PAT, chunk):
            word = word_match.group()
            chunk_token_ids.extend(self._encode_word(word))
        return chunk_token_ids
                        
    def _encode_word(self, word: str):
        """Encode a single word using BPE merges.
        
        Args:
            word: A single word to encode
            
        Returns:
            List of token IDs representing the word
            
        Example:
            # Encodes "hello" by applying BPE merges until no more merges are possible
            ids = self._encode_word("hello")
        """
        word_bytes = [bytes([b]) for b in word.encode("utf-8")]
        while True:
            pairs = [(i, (word_bytes[i], word_bytes[i+1])) for i in range(len(word_bytes)-1)]
            first_merged_pair = min(pairs, key=lambda x: self.merges_inverse.get(x[1], float("inf")), default=None)
            if not first_merged_pair or self.merges_inverse.get(first_merged_pair[1], float("inf")) == float("inf"):
                # we cannot find any more merges
                break
            
            merge_idx, (first_byte, second_byte) = first_merged_pair
            word_bytes[merge_idx] = first_byte + second_byte
            word_bytes.pop(merge_idx + 1)
            
        return [self.vocab_inverse[byte] for byte in word_bytes]
            
    def encode_iterable(self, iterable: Iterable[str], batch_size: int = 1000) -> Iterator[int]:
        """Encode an iterable of text strings into token IDs.
        
        Args:
            iterable: An iterable of text strings to encode
            batch_size: Number of items to process in each batch for efficiency
            
        Yields:
            Token IDs for all encoded text
        """
        batch = []
        for text in iterable:
            batch.append(text)
            if len(batch) >= batch_size:
                for item in batch:
                    yield from self.encode(item)
                batch = []
        
        for item in batch:
            yield from self.encode(item)
    
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded string
            
        Example:
            text = tokenizer.decode([1234, 5678, 9012])
            
        Raises:
            UnicodeDecodeError: If the bytes cannot be decoded to UTF-8
        """
        bytes_text = []
        for id in ids:
            bytes_text.append(self.vocab[id])
        return b"".join(bytes_text).decode("utf-8", errors="replace")