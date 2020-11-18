


string = ""
try:
    for line in iter(input,''):
        string+=line+'\n'
except EOFError:
    pass
print(string)