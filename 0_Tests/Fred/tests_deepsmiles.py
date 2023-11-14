import deepsmiles
print("DeepSMILES version: %s" % deepsmiles.__version__)
converter = deepsmiles.Converter(rings=True, branches=True)
print(converter) # record the options used

encoded = converter.encode("c1cccc(C(=O)Cl)c1")
print("Encoded: %s" % encoded)

try:
    decoded = converter.decode(encoded)
except deepsmiles.DecodeError as e:
    decoded = None
    print("DecodeError! Error message was '%s'" % e.message)

if decoded:
    print("Decoded: %s" % decoded)