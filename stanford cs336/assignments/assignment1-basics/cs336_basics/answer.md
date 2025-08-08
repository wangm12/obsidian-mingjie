# Problem1 

> What Unicode character does chr(0) return?
-> NULL

> How does this character’s string representation (__repr__()) differ from its printed representation?
-> __repr__() returns haxadecimal escape sequence; its unambiguous representation; shows exactly what characters are
printed representation: null is invisible; 
The key insight is:
repr() gives you a developer-friendly, unambiguous representation that shows '\x00'
print() attempts to display the actual character, which for null is nothing visible


---

# Problem 2

> What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings
-> 更小的文件尺寸和更高效的存储， 兼容性较好

> decode_utf8_bytes_to_str_wrong
-> It tries to decode individual characters into UTF-8; However, some characters may need multiple UTF-8 (variable-length encoding)

