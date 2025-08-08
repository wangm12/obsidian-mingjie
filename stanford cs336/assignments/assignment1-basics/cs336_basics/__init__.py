import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

test = chr(0)
print(test)
test = "this is a test" + chr(0) + "string"
print(test)
print("this is a test" + chr(0) + "string")