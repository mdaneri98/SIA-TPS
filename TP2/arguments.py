import json
import os
import configparser




class ProgramArguments():

    def __init__(self):
        self.arguments = {}
        self.init('arguments.config')


    def init(self, path):
        config = configparser.ConfigParser()
        config.read(path)
        self.arguments = {section: dict(config[section]) for section in config.sections()}


    def read_program_arguments(self, section, argument):
        return self.arguments[section][argument]
