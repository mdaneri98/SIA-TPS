from enum import Enum
from math import tanh
from typing import List, Tuple, Type, Any
import random

Character_type = Enum('type', ['Guerrero', 'Arquero', 'Defensor', 'Infiltrado'])
GenesOrder = ['ch_type', 'Height', 'Strength', 'Agility', 'Expertise', 'Resistance', 'Life']


class Item:

    def __init__(self, strength, agility, expertise, resistance, life):
        normalized_strength, normalized_agility, normalized_expertise, normalized_resistance, normalized_life = self._normalize_item(strength, agility, expertise, resistance, life)
        self.strength = normalized_strength
        self.agility = normalized_agility
        self.expertise = normalized_expertise
        self.resistance = normalized_resistance
        self.life = normalized_life

    @property
    def points(self):
        return self.strength + self.agility + self.expertise + self.resistance + self.life

    @staticmethod
    def _normalize_item(strength, agility, expertise, resistance, life):
        """ Normalizes the items to sum 150 """
        items = [strength, agility, expertise, resistance, life]
        total: float = 0.0
        for value in items:
            total += value

        if total == 150.0:
            return items

        factor: float = 150.0 / total
        for i, item in enumerate(items):
            items[i] *= factor

        return tuple(items)

    @staticmethod
    def create_item(strength, agility, expertise, resistance, life):
        (strength, agility, expertise, resistance, life) = Item._normalize_item(strength, agility, expertise,
                                                                                resistance, life)

        item = Item(strength, agility, expertise, resistance, life)
        return item

    @staticmethod
    def create_random_item():
        total_points = 150
        cuts = sorted(random.sample(range(1, total_points), 4))
        strength = cuts[0]
        agility = cuts[1] - cuts[0]
        expertise = cuts[2] - cuts[1]
        resistance = cuts[3] - cuts[2]
        life = total_points - cuts[3]
        return Item(strength, agility, expertise, resistance, life)

    def get_genes(self) -> list[Any]:
        return [self.strength, self.agility, self.expertise, self.resistance, self.life]

    @staticmethod
    def from_genes(genes: dict) -> 'Item':
        return Item(genes[0], genes[1], genes[2], genes[3], genes[4])


class Character:

    def __init__(self, ch_type: Character_type, height):
        self.ch_type = ch_type
        self._height = height
        self.items = []
        self.itemPoints = 0

    @property
    def attack(self):
        return (self.agility + self.expertise) * self.strength * self.attack_modifier

    @property
    def defense(self):
        return (self.resistance + self.expertise) * self.life * self.defense_modifier

    @property
    def attack_modifier(self):
        return 0.5 - pow(3 * self.height - 5, 4) + pow(3 * self.height - 5, 2) + self.height / 2

    @property
    def defense_modifier(self):
        return 2 + pow(3 * self.height - 5, 4) - pow(3 * self.height - 5, 2) - self.height / 2

    @property
    def strength(self):
        items_strength = sum(item.strength for item in self.items)
        return 100 * tanh(0.01 * items_strength)

    @property
    def agility(self):
        items_agility = sum(item.agility for item in self.items)
        return tanh(0.01 * items_agility)

    @property
    def expertise(self):
        items_expertise = sum(item.expertise for item in self.items)
        return 0.6 * tanh(0.01 * items_expertise)

    @property
    def resistance(self):
        items_resistance = sum(item.resistance for item in self.items)
        return tanh(0.01 * items_resistance)

    @property
    def life(self):
        items_life = sum(item.life for item in self.items)
        return 100 * tanh(0.01 * items_life)

    @property
    def height(self):
        return self._height

    @staticmethod
    def create_random_character(character_type):
        # random_ch_type = random.choice(list(Characterch_type))
        random_height = random.uniform(1.3, 2.0)
        ch = Character(character_type, random_height)

        item = Item.create_random_item()
        ch.add_item(item)

        return ch

    def add_item(self, item: Item) -> bool:
        self.items.append(item)
        self.itemPoints += item.points
        return True

    def performance(self):
        if self.ch_type.name == "Guerrero":
            return 0.6 * self.attack + 0.4 * self.defense
        elif self.ch_type.name == "Arquero":
            return 0.9 * self.attack + 0.1 * self.defense
        elif self.ch_type.name == "Defensor":
            return 0.1 * self.attack + 0.9 * self.defense
        elif self.ch_type.name == "Infiltrado":
            return 0.8 * self.attack + 0.3 * self.defense
        else:
            return "Tipo de personaje no válido."

    def get_genes(self) -> list:
        item_genes = [item.get_genes() for item in self.items]  # Múltiplo de 5
        genes = [self.ch_type.name, self.height]

        for item in self.items:
            genes.extend(item.get_genes())

        return genes

    @staticmethod
    def from_genes(genes: list) -> 'Character':
        ch = Character(Character_type[genes[0]], genes[1])

        for i in range(2, len(genes), 5):
            item_genes = genes[i:i + 5]
            item = Item(*item_genes)
            ch.add_item(item)

        return ch

    def __str__(self):
        return f"Character {self.ch_type} | Performance: {self.performance()} | Attack: {self.attack} | Defense: {self.defense} | Items points: {self.itemPoints}"
