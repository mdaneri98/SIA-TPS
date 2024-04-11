from enum import Enum
from math import tanh
from typing import List, Tuple, Type
import random

CharacterType = Enum('Type', ['Guerrero', 'Arquero', 'Defensor', 'Infiltrado'])
GenesOrder = ['Type', 'Height', 'Strength', 'Agility', 'Expertise', 'Resistance', 'Life']

class Item:

    def __init__(self, strength, agility, expertise, resistance, life):
        self.strength = strength
        self.agility = agility
        self.expertise = expertise
        self.resistance = resistance
        self.life = life

    @property
    def points(self):
        return self.strength + self.agility + self.expertise + self.resistance + self.life

    @staticmethod
    def create_item(strength, agility, expertise, resistance, life):
        points = strength + agility + expertise + resistance + life
        if (points > 150): # != ?
            return "Points exceed limit"
        item = Item(strength, agility, expertise, resistance, life)
        item.points = points
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

    def get_genes(self) -> dict:
        return [self.strength, self.agility, self.expertise, self.resistance, self.life]

    @staticmethod
    def from_genes(genes: dict) -> 'Item':
        return Item(genes[0], genes[1], genes[2], genes[3], genes[4])


class Character:

    def __init__(self, type: CharacterType, height):
        if 1.3 > height and height > 2:
            return "No valid height"
        self.type = type
        self.height = height
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
        return 0.5 - pow(3*self.height-5, 4) + pow(3*self.height-5, 2) + self.height/2
    
    @property
    def defense_modifier(self):
        return 2 + pow(3*self.height-5, 4) - pow(3*self.height-5, 2) - self.height/2
    
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

    @staticmethod
    def create_random_character(characterType):
        #random_type = random.choice(list(CharacterType))
        random_height = random.uniform(1.3, 2.0)
        ch = Character(characterType, random_height)
        
        item = Item.create_random_item()
        while (ch.add_item(item)):
            item = Item.create_random_item()

        return ch
        

    def add_item(self, item: Item) -> bool:
        if self.itemPoints + item.strength + item.agility + item.expertise + item.resistance + item.life > 150: 
            return False
        self.items.append(item)
        self.itemPoints += item.points
        return True


    def performance(self):
        if self.type.name == "Guerrero":
            return 0.6 * self.attack + 0.4 * self.defense
        elif self.type.name == "Arquero":
            return 0.9 * self.attack + 0.1 * self.defense
        elif self.type.name == "Defensor":
            return 0.1 * self.attack + 0.9 * self.defense
        elif self.type.name == "Infiltrado":
            return 0.8 * self.attack + 0.3 * self.defense
        else:
            return "Tipo de personaje no válido."


    def get_genes(self) -> list:
        item_genes = [item.get_genes() for item in self.items]  # Múltiplo de 5
        genes = [self.type.name, self.height]
        
        for item in self.items:
            genes.extend(item.get_genes())
        
        return genes


    @staticmethod
    def from_genes(genes: dict) -> 'Character':
        ch = Character(CharacterType[genes[0]], genes[1])

        for i in range(2, len(genes), 5):
            item_genes = genes[i:i+5]
            item = Item(*item_genes)
            ch.add_item(item)

        return ch



    def __str__(self):
        return f"Character {self.type} | Performance: {self.performance()} | Attack: {self.attack} | Defense: {self.defense} | Items points: {self.itemPoints}"