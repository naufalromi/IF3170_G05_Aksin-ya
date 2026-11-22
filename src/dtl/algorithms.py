import pandas as pd
import numpy as np
from collections import Counter
from pandas.api.types import is_numeric_dtype

from.dtl_utils import (
    EPSILON,
    MINIMUM_INSTANCES,
    estimate_error,
    search_best_attribute
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
            return self._error()
        else:
            e = 0
            for node in self.children:
                e += node._error
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
                metadata, branch_data, updated_weights, update, confidance_factor
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

