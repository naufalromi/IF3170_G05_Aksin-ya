import pandas as pd
import numpy as np
from collections import Counter
from pandas.api.types import is_numeric_dtype

from.dtl_utils import (
    EPSILON,
    MINIMUM_INSTANCES,
    estimate_error,
    search_best_attribute,
    _validate_weights
)

def branch_mask(column : pd.Series, operator : int, condition) -> pd.Series:
        """Mask baris yang mengikuti cabang (operator, condition), NaN selalu False"""
        if operator == Operator.EQUAL:
            return column == condition
        if operator == Operator.LESS_OR_EQUAL:
            return column <= condition
        return column > condition

class Operator:
    """Enum untuk melakukan operator
    """
    EQUAL = 1
    LESS_OR_EQUAL = 2
    GREATER = 3


class Node:
    """
        Class untuk mengambarkan decision Tree
    """
    
    def __init__(self, attribute : str, parent : Node = None, error : float = 0, total : float = 0, distribution = None):
        
        self.attribute = attribute
        self.parent = parent
        
        # private attribute
        self._error = error
        self._total = total
        self._distribution = distribution
        self._classes = None
        
        self.level = 0 if parent is None else parent.level + 1
        
        self.children = []
        self.conditions = []
        self.operation = []
    
    @property
    def classes(self):
        """Kembalikan daftar kelas yang dapat diprediksi oleh simpul (pohon) tersebut.
            hanya bisa root node
        """
        return self._classes
    
    @classes.setter
    def classes(self, classes):
        """Tetapkan daftar kelas yang dapat diprediksi oleh node (pohon) tersebut.
            Daftar ini digunakan untuk menentukan urutan probabilitas klasifikasi.
        """
        
        self._classes = classes
    
    def add(self, node : Node, condition : str | float, operator : Operator = Operator.EQUAL):
        """ Menambahkan ke child node"""
        
        node.parent = self
        self.children.append(node)
        self.conditions.append(condition)
        self.operation.append(operator)
    
    def to_text(self):
        self.__print_node("")
 
    
    def __print_node(self,prefix):
        """Print hasil dari pohon
        """

        if self.is_leaf():
            if self._error > 0:
                print("{} ({:.1f}/{:.1f})".format(self.attribute,
                                                  self._total,
                                                  self._error), end="")
            else:
                print("{} ({:.1f})".format(self.attribute,
                                           self._total), end="")
        else:
            if len(prefix) > 0:
                print("")

            for i, v in enumerate(self.conditions):
                if i > 0:
                    print("")

                operator = None

                if self.operation[i] == Operator.EQUAL:
                    operator = "="
                elif self.operation[i] == Operator.LESS_OR_EQUAL:
                    operator = "<="
                elif self.operation[i] == Operator.GREATER:
                    operator = ">"

                print("{}{} {} {}: ".format(
                    prefix, self.attribute, operator, v), end="")

                self.children[i].__print_node(prefix + "|    ")
                
    def total(self):
        if self.is_leaf():
            return self._total
        else:
            t = 0
            for node in self.children:
                t += node.total()
            return t
    
    def is_leaf(self):
        return len(self.conditions) == 0
    
    @staticmethod
    def __predict(instance: pd.DataFrame, node : Node, weight : np.array):
        probabilities = Counter()
        
        if node.is_leaf():
            for value, count in node._distribution.items():
                if count > 0:
                    probabilities[value] = weight * (count/node._total)
                
            if node._total == 0:
                probabilities[node.attribute] = weight
                
            return probabilities

        value = instance[node.attribute]
        
        if pd.isna(value):
            total = node.total()
            
            for i, v in enumerate(node.conditions):
                w = node.children[i].total() / total
                probabilities += Node.__predict(instance, node.children[i], weight * w)
        else:
            match = False
            
            for i, v in enumerate(node.conditions):
                if node.operation[i] == Operator.EQUAL and value == v:
                    match = True
                elif node.operation[i] == Operator.LESS_OR_EQUAL and value <= v:
                    match = True
                elif node.operation[i] == Operator.GREATER and value > v:
                    match = True
                
                if match:
                    probabilities += Node.__predict(instance, node.children[i], weight)
                    break
            # nilai missing atau kategori yang tidak pernah muncul saat training:
            # telusuri semua cabang, bobot sesuai proporsi data training di cabang tersebut
            if not match:
                total = node.total()
                
                for i, v in enumerate(node.conditions):
                    w = node.children[i].total()/total if total > 0 else 1 / len(node.children)
                    probabilities += node.__predict(instance, node.children[i], weight * w)
        return probabilities
    
    def predict(self, instance: pd.DataFrame):
        probabilities = Node.__predict(instance, self, 1.0)
        prediction = ("", 0)
        
        for value, count in probabilities.items():
            if count > prediction[1]:
                prediction = (value, count)
            
        if prediction[1] > 0:
            return prediction[0]

        raise Exception(
            f"Tidak bisa memprediksi nilai: nilai probabilitas:{str(dict(probabilities))}"
        )
    
    def probabilities(self, instance: pd.DataFrame):
        probabilities = Node.__predict(instance, self, 1.0)
        prediction = []
        
        for value in self.classes:
            prediction.append(probabilities[value])
        
        return prediction
    
    
    def error(self):
        """jumlah error di tarining. Internal node = Jumlah error anak"""
        
        if self.is_leaf():
            return self._error
        else:
            e = 0
            for node in self.children:
                e += node.error()
            return e
    
    def estimated(self, confidence_factor : float):
        """Estimasi error pesimis: error asli + penalti dari estimated_error"""
        
        if self.is_leaf():
            return self._error + estimate_error(self._total, self._error, confidence_factor)
        else:
            e = 0
            for node in self.children:
                e += node.estimated(confidence_factor)
            return e
    
    def adjust(self, data:pd.DataFrame):
        """Menganti threshold titik dengan dengan nilai terbesar di data yang <= threshold
            agar threshold adalah nilai yang benar dan sering muncul di training
        """
        if self.is_leaf():
            return
        
        operator = self.operation[0]
        threshold = self.conditions[0]
        
        if operator == Operator.LESS_OR_EQUAL or operator == Operator.GREATER:
            values = data[self.attribute].dropna().to_numpy()
            below = values[values <= threshold]
            if len(below) > 0:
                self.conditions = [below.max()] * len(self.conditions)
            
        for child in self.children:
            child.adjust(data)
    
    def estimate_with_data(self, metadata:dict, data : pd.DataFrame, weights: np.ndarray, update : bool = False, confidence_factor : float = 0.25):
        """Estimasi error pesimis subtree jika data 'data' dilewatkan ke subtree ini
           Jika update = true, statistik leaf (_total, _error, _distribution) diperbaharui"""
        
        if self.is_leaf():
            class_attribute = data.iloc[:, -1]
            total = weights.sum()
            correct = 0 if total == 0 else weights[class_attribute == self.attribute].sum()
            error = total - correct
            
            if update:
                distribution = Counter()
                for value in metadata[data.columns[-1]]:
                    distribution[value] =weights[class_attribute == value].sum()

                self._distribution = distribution
                self._total = total
                self._error = error
            return error + estimate_error(total,error,confidence_factor)
        
        missing = data[self.attribute].isna()
        known_weights = weights.sum() - weights[missing].sum()
        total = 0
        
        for i, v in enumerate(self.conditions):
            partition = branch_mask(data[self.attribute], self.operation[i], v)
            
            w = 0.0 if known_weights == 0 else weights[partition].sum()/known_weights
            
            updated_weights = weights.copy()
            updated_weights[missing] = updated_weights[missing] * w
            updated_weights = updated_weights[partition | missing]
            
            branch_data = data[partition | missing]
            if not is_numeric_dtype(data[self.attribute]):
                branch_data = branch_data.drop(columns=self.attribute)
            branch_data = branch_data.reset_index(drop = True)
            
            total += self.children[i].estimate_with_data(
                metadata, branch_data, updated_weights, update, confidence_factor
            )
        return total

    def sort(self):
        """Menaruh leaf didepan children untuk penampilan"""
        
        order = sorted(range(len(self.children)), key=lambda i: not self.children[i].is_leaf())
        self.children = [self.children[i] for i in order]
        self.conditions = [self.conditions[i] for i in order]
        self.operation = [self.operation[i] for i in order]
        
        for child in self.children:
            child.sort()


# bagian bikin pohon + pruning

def class_distribution(metadata : dict, data : pd.DataFrame, weights : np.ndarray) -> Counter:
    """Total bobot setiap kelas: {kelas : bobot}"""
    
    class_attribute = data.iloc[:, -1]
    distribution = Counter()
    for value in metadata[data.columns[-1]]:
        distribution[value] = weights[class_attribute == value].sum()
    return distribution

def calculate_majority(class_attribute : pd.Series, weights : np.ndarray):
    """Kelas dengan bobot terbesar. Jika seri ambil yang pertama muncul"""
    majority = None
    best = 0
    
    # implmenetasi asli pemilihan kalau sama di random tapi well jadi deterministik
    for value in class_attribute.unique():
        count = weights[class_attribute == value].sum()
        if count > best:
            majority = value
            best = count
    
    return majority

def make_leaf(metadata : dict, data : pd.DataFrame, weights : np.ndarray, majority, parent : Node) -> Node:
    """Membuat leaf yang memprediksi 'majority' """
    
    distribution = class_distribution(metadata, data, weights)
    total = weights.sum()
    error = total - distribution[majority] if total > 0 else 0
    return Node(majority, parent, error, total, distribution)

def pre_prune(metadata : dict, data:pd.DataFrame, majority, node:Node, weights:np.ndarray) -> Node:
    """Jika subtree tidak lebih dari leaf (error training), ganti dengan leaf"""
    length = weights.sum()
    if length > 0:
        majority = calculate_majority(data.iloc[:,-1], weights)
    
    leaf = make_leaf(metadata, data,weights, majority, node.parent)
    
    if node.error() >= leaf.error() - EPSILON:
        return leaf
    
    return node

def post_prune(metadata : dict, data : pd.DataFrame, node:Node, majority, weights : np.ndarray, confidence_factor : float = 0.25) -> Node:
    """Pessimistic error pruning C4.5. Membandingkan 3 pilihan
        1. subtree apa adanya
        2. diganti dengan leaf
        3. diganti dengan cabang terbesar
    """
    
    # 1. error subtree
    subtree_error = node.estimated(confidence_factor)
    
    # 2.  error jika jadi leaf
    leaf = make_leaf(metadata, data, weights, majority, node.parent)
    leaf_error = leaf.error() + estimate_error(leaf.total(),  leaf.error(), confidence_factor)
    
    # 3. error jika diganti cabang dengan data terbanyak
    selected : Node = max(node.children, key=lambda child:child.total())
    
    if selected.is_leaf():
        branch_error = leaf_error
    else:
        branch_error = selected.estimate_with_data(
            metadata, data,weights,confidence_factor=confidence_factor
        )
    
    if leaf_error <= subtree_error + 0.1 and leaf_error <= branch_error + 0.1:
        return leaf
    if branch_error <= subtree_error + 0.1:
        selected.estimate_with_data(metadata,data,weights,True,confidence_factor)
        selected.parent = node.parent
        return selected
    return node

def build_decision_tree(metadata : dict, data : pd.DataFrame, tree : Node = None, parent_majority= None, weights : np.ndarray = None, prune : bool = True, confidence_factor :float = 0.25 ) -> Node:
    """Membangung pohon C4.5 secara rekursif. kolom terakhir adalah data adalah kelas"""
    
    weights = _validate_weights(data, weights)
    
    class_attribute = data.iloc[:,-1]
    length = weights.sum()
    majority = parent_majority if length == 0 else calculate_majority(class_attribute, weights)
    
    # kasus basis : kelas sudah murni atau data terlalu sedikit
    if class_attribute.nunique() <= 1 or length < MINIMUM_INSTANCES * 2:
        return make_leaf(metadata, data, weights, majority, tree)
    
    attribute, info = search_best_attribute(data,weights)
    
    # kasus basis : tidak ada attribut yang memberi gain
    if attribute is None:
        return make_leaf(metadata, data, weights, majority,tree)
    
    column = data[attribute]
    missing = column.isna()
    known_weight = length - weights[missing].sum()
    numeric = is_numeric_dtype(column)
    
    if numeric :
        branches = [(Operator.LESS_OR_EQUAL,info[2]),(Operator.GREATER, info[2])]
    else:
        branches =[(Operator.EQUAL, value) for value in metadata[attribute]]
    
    # bobot setiap cabang (data missing dibagi proposional ke semua cabang)
    branch_weights = []
    for operator, condition in branches:
        count = weights[branch_mask(column,operator,condition)].sum()
        adjusted = count + weights[missing].sum() * (count/known_weight)
        branch_weights.append((count, adjusted))
    
    # minimal 2 cahan gharus punya data yang cukup, jika tidak attribut lain
    valid = sum(1 for _, adjusted in branch_weights if adjusted >= MINIMUM_INSTANCES)
    if valid < 2:
        return build_decision_tree(metadata, data.drop(columns=attribute), tree, parent_majority, weights,prune, confidence_factor)
    node = Node(attribute, parent=tree)
    
    for (operator, condition), (count, _) in zip(branches, branch_weights):
        partition = branch_mask(column,operator,condition)
        updated_weights = weights.copy()
        updated_weights[missing] = updated_weights[missing] * (count/known_weight)
        
        updated_weights = updated_weights[partition | missing]
        branch_data = data[partition | missing]
        if not numeric:
            # atribut kategorikal tidak dipakai lagi di bawahnya
            branch_data = branch_data.drop(columns = attribute)
        
        branch_data = branch_data.reset_index(drop = True)
        
        child = build_decision_tree(metadata, branch_data, node, majority,updated_weights,prune,confidence_factor)
        
        node.add(pre_prune(metadata,branch_data,majority,child,updated_weights),condition,operator)
        
    if prune:
        node = post_prune(metadata, data, node, majority, weights, confidence_factor)
    
    if tree is None:
        node.adjust(data)
        node.sort()
        node.classes = metadata[data.columns[-1]]
    
    return node
        
    
        
    