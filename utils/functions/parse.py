import re
import codecs
from typing import List
keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32', '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8', '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try', '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16', '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64', '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr', 'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register', 'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert', 'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq', 'NULL'})
main_set = frozenset({'main'})
main_args = frozenset({'argc', 'argv'})
operators3 = {'<<=', '>>='}
operators2 = {'->', '++', '--', '**', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='}
operators1 = {'(', ')', '[', ']', '.', '+', '&', '%', '<', '>', '^', '|', '=', ',', '?', ':', '{', '}', '!', '~'}

def to_regex(lst):
    return '|'.join([f'({re.escape(el)})' for el in lst])
regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)

def clean_gadget(gadget):
    fun_symbols = {}
    var_symbols = {}
    fun_count = 1
    var_count = 1
    rx_fun = re.compile('\\b([_A-Za-z]\\w*)\\b(?=\\s*\\()')
    rx_var = re.compile('\\b([_A-Za-z]\\w*)\\b((?!\\s*\\**\\w+))(?!\\s*\\()')
    cleaned_gadget = []
    for line in gadget:
        ascii_line = re.sub('[^\\x00-\\x7f]', '', line)
        hex_line = re.sub('0[xX][0-9a-fA-F]+', 'HEX', ascii_line)
        user_fun = rx_fun.findall(hex_line)
        user_var = rx_var.findall(hex_line)
        for fun_name in user_fun:
            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                if fun_name not in fun_symbols.keys():
                    fun_symbols[fun_name] = 'FUN' + str(fun_count)
                    fun_count += 1
                hex_line = re.sub('\\b(' + fun_name + ')\\b(?=\\s*\\()', fun_symbols[fun_name], hex_line)
        for var_name in user_var:
            if len({var_name[0]}.difference(keywords)) != 0 and len({var_name[0]}.difference(main_args)) != 0:
                if var_name[0] not in var_symbols.keys():
                    var_symbols[var_name[0]] = 'VAR' + str(var_count)
                    var_count += 1
                hex_line = re.sub('\\b(' + var_name[0] + ')\\b(?:(?=\\s*\\w+\\()|(?!\\s*\\w+))(?!\\s*\\()', var_symbols[var_name[0]], hex_line)
        cleaned_gadget.append(hex_line)
    return cleaned_gadget

def tokenizer(code, flag=False):
    gadget: List[str] = []
    tokenized: List[str] = []
    no_str_lit_line = re.sub('["]([^"\\\\\\n]|\\\\.|\\\\\\n)*["]', '', code)
    no_char_lit_line = re.sub("'.*?'", '', no_str_lit_line)
    code = no_char_lit_line
    if flag:
        decode_flag = True
        while decode_flag:
            try:
                code = codecs.getdecoder('unicode_escape')(no_char_lit_line)[0]
                decode_flag = False
            except UnicodeDecodeError as e:
                pos = e.start
                no_char_lit_line = no_char_lit_line[:pos] + no_char_lit_line[pos + 1:]
    for line in code.splitlines():
        if line == '':
            continue
        stripped = line.strip()
        gadget.append(stripped)
    clean = clean_gadget(gadget)
    for cg in clean:
        if cg == '':
            continue
        pat = re.compile('(/\\*([^*]|(\\*+[^*\\/]))*\\*+\\/)|(\\/\\/.*)')
        cg = re.sub(pat, '', cg)
        cg = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(\\r)', '', cg)
        splitter = ' +|' + regex_split_operators + '|(\\/)|(\\;)|(\\-)|(\\*)'
        cg = re.split(splitter, cg)
        cg = list(filter(None, cg))
        cg = list(filter(str.strip, cg))
        tokenized.extend(cg)
    return tokenized
