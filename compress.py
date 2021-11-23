"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
import time
from typing import Dict, Tuple, List
from utils import *
from huffman import HuffmanTree


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}
    for x in text:
        if x not in d:
            d[x] = 1
        else:
            d[x] += 1
    return d


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.
    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    sort_l = []
    d = freq_dict.copy()
    while d != {}:
        minimum = float('inf')
        key = 0
        for x in d:
            if d.get(x) < minimum:
                minimum = d.get(x)
                key = x
        sort_l.append((minimum, HuffmanTree(key)))
        del d[key]
    return _recursive_huffman(sort_l)


def _recursive_huffman(freq_lst: List[Tuple[int, HuffmanTree]]) -> HuffmanTree:
    """Does the recursive step of the tree by taking in a <freq_lst> and
    returns the huffman tree corresponding to the freq_list
    """
    lst = []
    copy_lst = freq_lst.copy()
    while copy_lst != []:
        freq = float('inf')
        tree = 0
        for x in copy_lst:
            if x[0] < freq:
                freq = x[0]
                tree = x[1]
        lst.append((freq, tree))
        if (freq, tree) in copy_lst:
            copy_lst.remove((freq, tree))

    if len(lst) == 0:
        return HuffmanTree(None, None, None)
    elif len(lst) == 1:
        single_value = lst[0][1].symbol
        dummy = (single_value + 1) % 256
        if dummy > single_value:
            return HuffmanTree(None, lst[0][1],
                               HuffmanTree(dummy))
        else:
            return HuffmanTree(None, HuffmanTree(dummy),
                               lst[0][1])
    elif len(lst) == 2:
        return HuffmanTree(None, lst[0][1], lst[1][1])
    else:
        sort_l = lst.copy()
        n1 = sort_l[0][1]
        n2 = sort_l[1][1]
        n1_key = sort_l[0][0]
        n2_key = sort_l[1][0]
        sort_l.remove(lst[0])
        sort_l.remove(lst[1])
        total = n1_key + n2_key
        tree = HuffmanTree(None, n1, n2)
        sort_l.append((total, tree))
        return _recursive_huffman(sort_l)


def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    if tree.is_leaf():
        return {}
    else:
        lst = _recursive_code(tree, '', '')
        d = {}
        for x in lst:
            d[x[0]] = x[1]
        return d


def _recursive_code(tree: HuffmanTree, addl: str, addr: str) \
        -> List[Tuple[int, str]]:
    """ Takes in the <tree> and the left branch <addl> and the right branch
    <addr> and returns a list of tuples
    """
    lst = []
    flag1 = True
    flag2 = True
    if tree.is_leaf():
        return []
    if tree.left.is_leaf():
        addl += '0'
        lst.append((tree.left.symbol, addl))
        flag2 = False
    else:
        addl += '0'
    if tree.right.is_leaf():
        addr += '1'
        lst.append((tree.right.symbol, addr))
        flag1 = False
    else:
        addr += '1'

    if not flag1 and not flag2:
        return lst
    elif flag1 and not flag2:
        var = lst + _recursive_code(tree.right, addr, addr)
        return var
    elif flag2 and not flag1:
        var = lst + _recursive_code(tree.left, addl, addl)
        return var
    else:
        var = lst + _recursive_code(tree.left, addl, addl) + \
              _recursive_code(tree.right, addr, addr)
        return var


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to post order traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    if tree:
        _number_nodes_helper(tree, 0)


def _number_nodes_helper(tree: HuffmanTree, filler: int) -> None:
    """ numbers the nodes of the <tree> in post order traversal. and uses
    <filler> to keep track of the number
    """
    if not tree.left and tree.right:
        if tree.right.is_leaf():
            tree.number = filler
            filler += 1
        else:
            _number_nodes_helper(tree.right, filler)
            filler = tree.right.number + 1
            tree.number = filler
            filler += 1

    elif not tree.right and tree.left:
        if tree.left.is_leaf():
            tree.number = filler
            filler += 1
        else:
            _number_nodes_helper(tree.left, filler)
            filler = tree.left.number + 1
            tree.number = filler
            filler += 1

    elif tree.left and tree.right:
        if tree.left.is_leaf() and tree.right.is_leaf():
            tree.number = filler
            filler += 1
        if not (tree.right.is_leaf() or tree.left.is_leaf()):
            _number_nodes_helper(tree.left, filler)
            filler = tree.left.number + 1
            _number_nodes_helper(tree.right, filler)
            filler = tree.right.number + 1
            tree.number = filler
            filler += 1
        if not tree.right.is_leaf() and tree.left.is_leaf():
            _number_nodes_helper(tree.right, filler)
            filler = tree.right.number + 1
            tree.number = filler
            filler += 1
        if not tree.left.is_leaf() and tree.right.is_leaf():
            _number_nodes_helper(tree.left, filler)
            filler = tree.left.number + 1
            tree.number = filler
            filler += 1


def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    num = 0
    den = 0
    codes = get_codes(tree)
    for key in freq_dict:
        den += freq_dict.get(key, 0)
        num += len(codes.get(key)) * freq_dict.get(key, 0)
    if den == 0:
        return num / 1
    else:
        return num / den


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    code = ''
    for x in text:
        code += codes.get(x)
    end = 1
    temp = ''
    lst = []
    for y in code:
        temp += y
        if end == 8:
            lst.append(temp)
            temp = ''
            end = 0
        end += 1
    if temp:
        lst.append(temp)
    if not lst:
        return bytes([])
    while len(lst[-1]) != 8:
        lst[-1] += '0'
    ans = []
    d = {}
    for z in lst:
        if z not in d:
            d[z] = bits_to_byte(z)
        if z in d:
            ans.append(d.get(z))
        # ans.append(bits_to_byte(z))
    return bytes(ans)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    ans = []
    flag_left = False
    flag_right = False
    if not tree.is_leaf():
        if tree.left.is_leaf():
            ans.append(0)
            ans.append(tree.left.symbol)
            flag_left = True
        else:
            ans.append(1)
            ans.append(tree.left.number)
        if tree.right.is_leaf():
            ans.append(0)
            ans.append(tree.right.symbol)
            flag_right = True
        else:
            ans.append(1)
            ans.append(tree.right.number)
    else:
        return b''

    if flag_left and flag_right:
        return bytes(ans)
    elif not flag_left and not flag_right:
        var = list(tree_to_bytes(tree.left)) + list(tree_to_bytes(tree.right)) \
              + ans
        return bytes(var)
    elif not flag_left and flag_right:
        var = list(tree_to_bytes(tree.left)) + ans
        return bytes(var)
    elif not flag_right and flag_left:
        var = list(tree_to_bytes(tree.right)) + ans
        return bytes(var)
    return None


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """

    if not node_lst:
        return HuffmanTree(None, None, None)
    elif node_lst[root_index].l_type == 0 and node_lst[root_index].r_type == 0:
        return HuffmanTree(None, HuffmanTree(node_lst[root_index].l_data),
                           HuffmanTree(node_lst[root_index].r_data))
    else:
        if node_lst[root_index].l_type == 1 and \
                node_lst[root_index].r_type == 0:
            return HuffmanTree(None,
                               generate_tree_general(node_lst, node_lst[
                                   root_index].l_data),
                               HuffmanTree(node_lst[root_index].r_data))
        elif node_lst[root_index].l_type == 0 and \
                node_lst[root_index].r_type == 1:
            return HuffmanTree(None, HuffmanTree(node_lst[root_index].l_data),
                               generate_tree_general(node_lst, node_lst[
                                   root_index].r_data))
        elif node_lst[root_index].l_type == 1 and \
                node_lst[root_index].r_type == 1:
            return HuffmanTree(None,
                               generate_tree_general(node_lst, node_lst[
                                   root_index].l_data),
                               generate_tree_general(node_lst, node_lst[
                                   root_index].r_data))
        return None


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    root_index = len(node_lst)
    lst = node_lst[:root_index]
    return _postorder_helper(lst)[0]


def _postorder_helper(lst: List[ReadNode]) -> \
        Tuple[HuffmanTree, List[ReadNode]]:
    """ Takes in <lst> a list of ReadNode and outputs
    a tuple of the tree and the remaining version of the <lst>
    """
    if not lst:
        return HuffmanTree(), []
    temp = lst.pop()
    if temp.l_type == 0 and temp.r_type == 0:
        right = HuffmanTree(temp.l_data)
        left = HuffmanTree(temp.r_data)
        return HuffmanTree(None, right, left), lst
    elif temp.l_type == 0 and temp.r_type == 1:
        right = _postorder_helper(lst)[0]
        left = HuffmanTree(temp.l_data)
        return HuffmanTree(None, left, right), lst
    elif temp.l_type == 1 and temp.r_type == 0:
        right = HuffmanTree(temp.r_data)
        left = _postorder_helper(lst)[0]
        return HuffmanTree(None, left, right), lst
    elif temp.l_type == 1 and temp.r_type == 1:
        right, lst2 = _postorder_helper(lst)
        left = _postorder_helper(lst2)[0]
        return HuffmanTree(None, left, right), lst
    else:
        return HuffmanTree(), []


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    string = ''
    d = {}
    for byte in text:
        if byte not in d:
            d[byte] = byte_to_bits(byte)
        if byte in d:
            string += d.get(byte)
    # string = temp.join(byte_to_bits(byte) for byte in text)
    codes = get_codes(tree)
    inv_code = {v: k for k, v in codes.items()}
    acc = ''
    ans = []
    for x in string:
        acc += x
        if acc in inv_code:
            ans.append(inv_code.get(acc))
            acc = ''
    return bytes(ans[:size])


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    lst = _create_lst(tree, 0)
    sorted_lst = _sort_lst(lst[:], freq_dict)
    _swapper(lst, sorted_lst, tree)


def _create_lst(tree: HuffmanTree, c: int) -> List[Tuple[int, int]]:
    """ Creates a list of tuples with the leafs and the depth using the
    <tree> and a depth counter <c>
    """
    c += 1
    if tree.is_leaf():
        return [(tree.symbol, c)]
    elif not tree.is_leaf():
        lst = []
        lst += _create_lst(tree.left, c)
        c = lst[-1][1]
        lst += _create_lst(tree.right, c)
        return lst
    else:
        return None


def _sort_lst(lst: List[Tuple[int, int]], freq_dict: Dict[int, int]) -> \
        List[int]:
    """ sorts the tuple list <lst> and uses <freq_dict> and returns an
    output of all the leafs in order from highest freq to lowest freq
    """
    new_lst = []
    while len(lst) != 0:
        maximum = float('-inf')
        for x in lst:
            if freq_dict.get(x[0]) > maximum:
                maximum = freq_dict.get(x[0])
                insert = x
        new_lst.append(insert[0])
        lst.remove(insert)
    return new_lst


def _swapper(lst: List[Tuple[int, int]], sorted_lst: List[int],
             tree: HuffmanTree) -> None:
    """Takes <lst> and <sorted_lst> and the <tree> and swaps the values if there
    needs to be swapping
    """
    if tree.is_leaf():
        for x in lst:
            if x[0] == tree.symbol:
                tree.symbol = sorted_lst[0]
                sorted_lst.remove(tree.symbol)
                break

    else:
        _swapper(lst, sorted_lst, tree.left)
        _swapper(lst, sorted_lst, tree.right)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input("Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))
