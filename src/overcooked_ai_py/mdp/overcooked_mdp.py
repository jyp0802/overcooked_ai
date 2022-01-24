import itertools, copy, warnings
import numpy as np
import yaml, os
import threading
from functools import reduce
from collections import defaultdict, Counter
from overcooked_ai_py.utils import pos_distance, read_layout_dict, classproperty
from overcooked_ai_py.mdp.actions import Action, Direction

def mydebug(msg):
    if type(msg) == list:
        msg = ", ".join([str(x) for x in msg])
    print("!!", msg)

'''
TODO:
* change .format to f'strings'
* separate game logic and RL logic
* add 'activate' action

ingredients:
- tomato, onion, mushroom
- bread, cheese, meat
- rice, seaweed, fish, cucumber
- flour, chocolate, strawberry
- pasta, shrimp
- potato, chicken

utensil:
- pot, cutting board
- frying pan
- mixer, oven
- fryer

serving:
- plate, sink
'''

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
config = yaml.safe_load(open(os.path.join(cur_file_dir, 'my_config.yaml'), 'r'))


CFG_CONTAINER_INFO = config['container_info']
CFG_RECIPE = config['recipe_info']
CFG_RECIPE_12_3 = {(a, b): c for (a, b, c) in CFG_RECIPE}
CFG_RECIPE_3_12 = {c: (a, b) for (a, b, c) in CFG_RECIPE}
# JYP: Need to fix for the case where the same ingredients lead
# to different outcomes depending on the container used.
CFG_RECIPE_1_23 = {a: (b, c) for (a, b, c) in CFG_RECIPE}

CFG_ALL_RAWFOOD = config['raw_foods']
CFG_ALL_INGREDIENTS = list(set(CFG_ALL_RAWFOOD + list(CFG_RECIPE_3_12.keys())))
CFG_NUM_INGREDIENT_TYPE = len(CFG_ALL_INGREDIENTS)
CFG_MAX_NUM_INGREDIENTS = config['max_num_ingredients']

CFG_ALL_CONTAINERS = config['containers']
CFG_DELIVERABLE_CONTAINER = config['deliverable_containers']

CFG_ALL_DISPENSERS = config['dispensers']

CFG_ALL_OBJECTS = CFG_ALL_RAWFOOD + CFG_ALL_CONTAINERS

CFG_ALL_TERRAINS = config['terrains']

CFG_DEFAULT_OBJECTS = config['default_objects']

CFG_STR_REP = config['object_representation']

CFG_TERRAIN_TO_SYMBOL = config['terrain_to_symbol']
CFG_SYMBOL_TO_TERRAIN = {v: k for k, v in CFG_TERRAIN_TO_SYMBOL.items()}

CFG_POTENTIAL_CONSTANTS = config['potential_constants']
CFG_DEFAULT_POTENTIAL = config['default_potential']
CFG_DEFAULT_RECIPE_VALUE = config['default_recipe_value']
CFG_DEFAULT_RECIPE_TIME = config['default_recipe_time']

EVENT_TYPES = []
for elem in CFG_ALL_OBJECTS:
    EVENT_TYPES += [f"{elem}_pick", f"{elem}_drop", f"useful_{elem}_pick", f"useful_{elem}_drop"]

for cont in CFG_ALL_CONTAINERS:
    for ingr in CFG_ALL_INGREDIENTS:
        EVENT_TYPES.append(f"{ingr}_to_{cont}")
        EVENT_TYPES += [f"{state}_{ingr}_to_{cont}" for state in ["optimal", "catastrophic", "viable", "useless"]]

EVENT_TYPES.append("deliver")

BASE_REW_SHAPING_PARAMS = config['BASE_REW_SHAPING_PARAMS']

class Recipe:
    ALL_RECIPES_CACHE = {}

    _computed = False
    _configured = False
    _conf = {}
    
    def __new__(cls, ingredients):
        if not cls._configured:
            raise ValueError("Recipe class must be configured before recipes can be created")
        # Some basic argument verification
        if not ingredients or not hasattr(ingredients, '__iter__') or len(ingredients) == 0:
            raise ValueError("Invalid input recipe. Must be ingredients iterable with non-zero length")
        for elem in ingredients:
            if not elem in CFG_ALL_INGREDIENTS:
                raise ValueError("Invalid ingredient: {0}. Recipe can only contain ingredients {1}".format(elem, CFG_ALL_INGREDIENTS))
        if not len(ingredients) <= CFG_MAX_NUM_INGREDIENTS:
            raise ValueError("Recipe of length {0} is invalid. Recipe can contain at most {1} ingredients".format(len(ingredients), CFG_MAX_NUM_INGREDIENTS))
        key = hash(tuple(sorted(ingredients)))
        if key in cls.ALL_RECIPES_CACHE:
            return cls.ALL_RECIPES_CACHE[key]
        cls.ALL_RECIPES_CACHE[key] = super(Recipe, cls).__new__(cls)
        return cls.ALL_RECIPES_CACHE[key]

    def __init__(self, ingredients):
        self._ingredients = ingredients
        # self.food = food
        # self.ingredients, self.action = CFG_RECIPE_3_12[food]
        # self.ingredients = self.ingredients.split(", ")

    def __getnewargs__(self):
        assert False
        return (self._ingredients,)

    def __int__(self):
        ingredient_count = []
        for elem in CFG_ALL_INGREDIENTS:
            ingredient_count.append(self.ingredients.count(elem))

        mixed_mask = int(bool(np.prod(ingredient_count)))
        mixed_shift = (CFG_MAX_NUM_INGREDIENTS + 1)**CFG_NUM_INGREDIENT_TYPE
        encoding = 0
        for idx, cnt in enumerate(ingredient_count):
            encoding += (CFG_MAX_NUM_INGREDIENTS + 1) ** idx * cnt

        return mixed_mask * encoding * mixed_shift + encoding

    def __hash__(self):
        return hash(self.ingredients)

    def __eq__(self, other):
        # The ingredients property already returns sorted items, so equivalence check is sufficient
        return self.ingredients == other.ingredients

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __ge__(self, other):
        return int(self) >= int(other)

    def __repr__(self):
        return self.ingredients.__repr__()

    def __iter__(self):
        assert False
        return iter(self.ingredients)

    def __copy__(self):
        return Recipe(self.ingredients)

    def __deepcopy__(self, memo):
        ingredients_cpy = copy.deepcopy(self.ingredients)
        return Recipe(ingredients_cpy)

    @classmethod
    def _compute_all_recipes(cls):
        for i in range(CFG_MAX_NUM_INGREDIENTS):
            for ingredient_list in itertools.combinations_with_replacement(CFG_ALL_INGREDIENTS, i + 1):
                cls(ingredient_list)

    @property
    def ingredients(self):
        return tuple(sorted(self._ingredients))

    @ingredients.setter
    def ingredients(self, _):
        raise AttributeError("Recipes are read-only. Do not modify instance attributes after creation")

    @property
    def value(self):
        if self._delivery_reward:
            return self._delivery_reward
        if self._value_mapping and self in self._value_mapping:
            return self._value_mapping[self]
        if all(v is None for v in self._ingredient_value.values()):
            all_value = 0
            stack = self._ingredients
            while stack:
                cur_food = stack.pop()
                if self._ingredient_value[cur_food] != None:
                    all_value += self._ingredient_value[cur_food]
                elif cur_food in CFG_RECIPE_3_12:
                    sub_foods, cont = CFG_RECIPE_3_12[cur_food]
                    sub_foods = sub_foods.split(", ")
                    stack += sub_foods
                    all_time += CFG_CONTAINER_INFO[cont]['cook_time']
            self._delivery_reward = all_value
            return self._delivery_reward
        return CFG_DEFAULT_RECIPE_VALUE

    @property
    def time(self):
        if self._cook_time:
            return self._cook_time
        if self._time_mapping and self in self._time_mapping:
            return self._time_mapping[self]
        all_time = 0
        stack = self._ingredients
        while stack:
            cur_food = stack.pop()
            if cur_food in CFG_RECIPE_3_12:
                sub_foods, cont = CFG_RECIPE_3_12[cur_food]
                sub_foods = sub_foods.split(", ")
                stack += sub_foods
                all_time += CFG_CONTAINER_INFO[cont]['cook_time']
        self._cook_time = all_time
        return self._cook_time

    def to_dict(self):
        return { 'ingredients' : self.ingredients }

    def neighbors(self):
        """
        Return all "neighbor" recipes to this recipe. A neighbor recipe is one that can be obtained
        by either adding exactly one ingredient to the current recipe or performing one extra action
        on (a subset of) the ingredients of the current recipe
        """
        neighbors = []
        if len(self.ingredients) < CFG_MAX_NUM_INGREDIENTS:
            for ingredient in CFG_ALL_INGREDIENTS:
                new_ingredients = [*self.ingredients, ingredient]
                new_recipe = Recipe(new_ingredients)
                neighbors.append(new_recipe)
        for sub_len in range(len(self.ingredients)+1):
            for sub_ingr in itertools.combinations(self.ingredients, sub_len):
                if sub_ingr in CFG_RECIPE_1_23:
                    new_ingr = list(self.ingredients)
                    for x in sub_ingr:
                        new_ingr.remove(x)
                    new_ingr += CFG_RECIPE_1_23[sub_ingr][2]
                    new_recipe = Recipe(new_ingr)
                    neighbors.append(new_recipe)
        return neighbors

    @classproperty
    def ALL_RECIPES(cls):
        if not cls._computed:
            cls._compute_all_recipes()
            cls._computed = True
        return set(cls.ALL_RECIPES_CACHE.values())

    @classproperty
    def configuration(cls):
        if not cls._configured:
            raise ValueError("Recipe class not yet configured")
        return cls._conf

    @classmethod
    def configure(cls, conf):
        global CFG_MAX_NUM_INGREDIENTS
        cls._conf = conf
        cls._configured = True
        cls._computed = False
        if 'max_num_ingredients' in conf:
            CFG_MAX_NUM_INGREDIENTS = conf.get('max_num_ingredients')

        cls._cook_time = None
        cls._delivery_reward = None
        cls._value_mapping = None
        cls._time_mapping = None
        cls._ingredient_value = {elem: None for elem in CFG_ALL_INGREDIENTS}

        ## Basic checks for validity ##

        # Mutual Exclusion
        if 0 < len([_ for _ in conf.keys() if "_ingr_time" in _]) < CFG_NUM_INGREDIENT_TYPE:
            raise ValueError("Must specify times for all ingredients")
        if 0 < len([_ for _ in conf.keys() if "_ingr_value" in _]) < CFG_NUM_INGREDIENT_TYPE:
            raise ValueError("Must specify values for all ingredients")

        if '_ingr_value' in '\t'.join(conf.keys()) and 'delivery_reward' in conf:
            raise ValueError("'delivery_reward' incompatible with '<ingredient>_ingvalue'")
        if '_ingr_value' in '\t'.join(conf.keys()) and 'recipe_values' in conf:
            raise ValueError("'recipe_values' incompatible with '<ingredient>_ingvalue'")
        if 'recipe_values' in conf and 'delivery_reward' in conf:
            raise ValueError("'delivery_reward' incompatible with 'recipe_values'")
        if '_ingr_time' in '\t'.join(conf.keys()) and 'cook_time' in conf:
            raise ValueError("'cook_time' incompatible with '<ingredient>_ingr_time")
        if '_ingr_time' in '\t'.join(conf.keys()) and 'recipe_times' in conf:
            raise ValueError("'recipe_times' incompatible with '<ingredient>_ingr_time'")
        if 'recipe_times' in conf and 'cook_time' in conf:
            raise ValueError("'cook_time' incompatible with 'recipe_times'")

        # recipe_ lists and orders compatibility
        if 'recipe_values' in conf:
            if not 'all_orders' in conf or not conf['all_orders']:
                raise ValueError("Must specify 'all_orders' if 'recipe_values' specified")
            if not len(conf['all_orders']) == len(conf['recipe_values']):
                raise ValueError("Number of recipes in 'all_orders' must be the same as number in 'recipe_values")
        if 'recipe_times' in conf:
            if not 'all_orders' in conf or not conf['all_orders']:
                raise ValueError("Must specify 'all_orders' if 'recipe_times' specified")
            if not len(conf['all_orders']) == len(conf['recipe_times']):
                raise ValueError("Number of recipes in 'all_orders' must be the same as number in 'recipe_times")

        
        ## Conifgure ##

        if 'cook_time' in conf:
            cls._cook_time = conf['cook_time']

        if 'delivery_reward' in conf:
            cls._delivery_reward = conf['delivery_reward']

        if 'recipe_values' in conf:
            assert "all_orders has changed from dict to recipe food name" == ""
            cls._value_mapping = {
                cls.from_dict(recipe) : value for (recipe, value) in zip(conf['all_orders'], conf['recipe_values'])
            }

        if 'recipe_times' in conf:
            assert "all_orders has changed from dict to recipe food name" == ""
            print('bye', conf['all_orders'], conf['recipe_times'])
            cls._time_mapping = {
                cls.from_dict(recipe) : time for (recipe, time) in zip(conf['all_orders'], conf['recipe_times'])
            }

        for elem in CFG_ALL_INGREDIENTS:
            if f"{elem}_ingr_value" in conf:
                cls._ingredient_value[elem] = conf[f"{elem}_ingr_value"]
    
    @classmethod
    def generate_random_recipes(cls, n=1, min_size=2, max_size=3, ingredients=None, recipes=None, unique=True):
        """
        n (int): how many recipes generate
        min_size (int): min generated recipe size
        max_size (int): max generated recipe size
        ingredients (list(str)): list of ingredients used for generating recipes (default is CFG_ALL_INGREDIENTS)
        recipes (list(Recipe)): list of recipes to choose from (default is cls.ALL_RECIPES)
        unique (bool): if all recipes are unique (without repeats)
        """
        if recipes is None: recipes = cls.ALL_RECIPES

        ingredients = set(ingredients or CFG_ALL_INGREDIENTS)
        choice_replace = not(unique)

        assert 1 <= min_size <= max_size <= CFG_MAX_NUM_INGREDIENTS
        assert all(ingredient in CFG_ALL_INGREDIENTS for ingredient in ingredients)

        def valid_size(r):
            return min_size <= len(r.ingredients) <= max_size

        def valid_ingredients(r):
            return all(i in ingredients for i in r.ingredients)
        
        relevant_recipes = [r for r in recipes if valid_size(r) and valid_ingredients(r)]
        assert choice_replace or (n <= len(relevant_recipes))
        return np.random.choice(relevant_recipes, n, replace=choice_replace)

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)

    @classmethod
    def get_result_food_name(cls, ingredient_names, container_name):
        ingredient_str = ", ".join(sorted(ingredient_names))
        if (ingredient_str, container_name) not in CFG_RECIPE_12_3:
            raise ValueError(f"Recipe for {ingredient_str} and {container_name} doesn't exist in config!")
        return CFG_RECIPE_12_3[(ingredient_str, container_name)]
        

class FoodState(object):
    """
    State of an ingredient in OvercookedGridworld.
    """

    def __init__(self, name, position, **kwargs):
        """
        name (str): The name of the object
        position (int, int): Tuple for the current location of the object.
        """
        self.name = name
        self._position = tuple(position)
        self._recipe = None

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_pos):
        self._position = new_pos

    def is_valid(self):
        return self.name in CFG_ALL_INGREDIENTS

    def deepcopy(self):
        return FoodState(self.name, self.position)

    def __eq__(self, other):
        return isinstance(other, FoodState) and \
            self.name == other.name and \
            self.position == other.position

    def __hash__(self):
        return hash((self.name, self.position))

    def __repr__(self):
        return '{}@{}'.format(
            self.name, self.position)

    def __str__(self):
        return CFG_STR_REP[self.name]

    def to_dict(self):
        return {
            "name": self.name,
            "position": self.position
        }

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return FoodState(**obj_dict)

    @property
    def recipe(self):
        if self.is_idle:
            raise ValueError("Recipe is not determined until soup begins cooking")
        if not self._recipe:
            self._recipe = Recipe(self.name)
        return self._recipe


class ContainerState(object):
    """
    State of a container in OvercookedGridworld.
    """

    def __init__(self, name, position, ingredients=[], cooking_tick=-1, **kwargs):
        """
        name (str): The name of the object
        position (int, int): Tuple for the current location of the object.
        """
        assert name in CFG_CONTAINER_INFO
        self.name = name
        self._position = tuple(position)
        self._ingredients = ingredients
        self._cooking_tick = cooking_tick
        self._recipe = None
        self.cook_time = CFG_CONTAINER_INFO[name]['cook_time']
        if self.cook_time == 0:
            self._cooking_tick = 0
        self.max_ingredients = CFG_CONTAINER_INFO[name]['max_ingredients']
        assert len(self._ingredients) <= self.max_ingredients

    def __eq__(self, other):
        return isinstance(other, ContainerState) and self.name == other.name and \
            self.position == other.position and self.max_ingredients == other.max_ingredients and \
            all([this_i == other_i for this_i, other_i in zip(self._ingredients, other._ingredients)])

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        return hash((self.name, self.position, ingredient_hash, self.max_ingredients))

    def __repr__(self):
        ingredients_str = self._ingredients.__repr__()
        return "{}@{}\nIngredients:\t{}\nCooking Tick:\t{}" \
            .format(self.name, self.position, ingredients_str, self._cooking_tick)

    def __str__(self):
        res = "{"
        for ingredient in sorted(self.ingredients):
            res += CFG_STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        res += "}"
        return res

    @property
    def position(self):
        return self._position

    @FoodState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def cook_time_remaining(self):
        return max(0, self.cook_time - self._cooking_tick)

    @property
    def is_ready(self):
        if self.is_idle or len(self._ingredients) == 0:
            return False
        return self._cooking_tick >= self.cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def is_full(self):
        return not self.is_idle or len(self.ingredients) == self.max_ingredients

    @property
    def is_empty(self):
        return len(self.ingredients) == 0

    @property
    def result_food_name(self):
        if not self.is_ready:
            raise ValueError("Cooking has not finished yet")
        # JYP: may need to change to if self.cook_time == 0
        if self.name == "dish" and len(self.ingredients) == 1:
            return self.ingredients[0].name
        return Recipe.get_result_food_name(self.ingredients, self.name)

    def is_valid(self):
        if not self.name in CFG_ALL_CONTAINERS:
            return False
        if not all([ingredient.position == self.position for ingredient in self._ingredients]):
            return False
        if len(self.ingredients) > self.max_ingredients:
            return False
        return True

    def can_add(self, food_name):
        # JYP: may need to check for the recipe that has the ingredient that this container already has
        # e.g., cont - flour, chocolate, milk
        #            - egg, rice
        # if container has rice, can't add flour
        if self.is_full:
            return False
        # JYP: might need to activate this `if` again
        # if food_name in self.ingredients:
        #     return False
        if food_name not in CFG_CONTAINER_INFO[self.name].get('can_add', []):
            return False
        return True

    def add_ingredient(self, ingredient):
        if not ingredient.name in CFG_ALL_INGREDIENTS:
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in this container")
        ingredient.position = self.position
        self._ingredients.append(ingredient)
        # JYP: Let's change it so that cooking starts once a button is pressed
        # if self.is_idle:
        #     self.begin_cooking()
        # else:
        #     self._cooking_tick /= 2

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = FoodState(ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this container at this time")
        if len(self.ingredients) == 0:
            raise ValueError("Must add at least one ingredient to container before you can begin cooking")
        self._cooking_tick = 0

    def cook(self):
        print('Cook')
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a container that is already done")
        if self._cooking_tick < self.cook_time:
            self._cooking_tick += 1
            print("cooking:", self._cooking_tick)

    def get_result_food(self):
        # JYP: may need to change to if self.cook_time == 0
        if self.name == "dish" and len(self.ingredients) == 1:
            result_food = self.ingredients[0]
        else:
            result_food = FoodState(self.result_food_name, self.position)
        self._ingredients = []
        self._cooking_tick = -1
        return result_food

    def deepcopy(self):
        return ContainerState(self.name, self.position, [ingredient.deepcopy() for ingredient in self._ingredients], self._cooking_tick)

    def to_dict(self):
        info_dict = {}
        info_dict['name'] = self.name
        info_dict['position'] = self.position
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict['_ingredients'] = ingrdients_dict
        info_dict['cooking_tick'] = self._cooking_tick
        info_dict['is_cooking'] = self.is_cooking
        info_dict['is_ready'] = self.is_ready
        info_dict['is_idle'] = self.is_idle
        info_dict['cook_time'] = -1 if self.is_idle else self.cook_time
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        assert obj_dict['name'] in CFG_ALL_CONTAINERS

        if 'state' in obj_dict:
            print('WHAT???')
            assert False
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict['state']
            cooking_tick = -1 if time == 0 else time
            finished = time >= 20
            ingredient_count = {ingredient: num_ingredient}
            return ContainerState.get_container(obj_dict['position'], ingredient_count, cooking_tick=cooking_tick, finished=finished)

        ingredients_objs = [FoodState.from_dict(ing_dict) for ing_dict in obj_dict['_ingredients']]
        obj_dict['ingredients'] = ingredients_objs
        return cls(**obj_dict)

    @classmethod
    def get_container(cls, name, position, ingredient_count, cooking_tick=-1, finished=False, **kwargs):
        num_ingredients = sum(ingredient_count.values())
        if num_ingredients < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_ingredients > self.max_ingredients:
            raise ValueError("Too many ingredients specified for this container")
        if cooking_tick >= 0 and num_ingredients == 0:
            raise ValueError("_cooking_tick must be -1 for empty container")
        if finished and num_ingredients == 0:
            raise ValueError("Empty container cannot be finished")
        
        ingredients = []
        for elem, num_elem in ingredient_count.items():
            ingredients += [FoodState(elem, position) for _ in range(num_elem)]

        container = cls(position, ingredients, cooking_tick)
        if finished:
            container._cooking_tick = container.cook_time
        return container
        

class PlayerState(object):
    """
    State of a player in OvercookedGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: FoodState or ContainerState representing the object held by the player, or
                 None if there is no such object.
    """
    def __init__(self, position, orientation, held_object=None):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, FoodState) or isinstance(self.held_object, ContainerState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return (self.position, self.orientation)

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj
 
    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj
    
    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(self.position, self.orientation, new_obj)

    def __eq__(self, other):
        return isinstance(other, PlayerState) and \
            self.position == other.position and \
            self.orientation == other.orientation and \
            self.held_object == other.held_object

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return '{} facing {} holding {}'.format(
            self.position, self.orientation, str(self.held_object))
    
    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": self.held_object.to_dict() if self.held_object is not None else None
        }

    @staticmethod
    def from_dict(player_dict):
        # JYP check if this is ever called
        assert False
        player_dict = copy.deepcopy(player_dict)
        held_obj = player_dict.get("held_object", None)
        if held_obj is not None:
            if held_obj["name"] in CFG_ALL_CONTAINERS:
                player_dict["held_object"] = ContainerState.from_dict(held_obj)
            elif held_obj["name"] in CFG_ALL_RAWFOOD:
                player_dict["held_object"] = FoodState.from_dict(held_obj)
            else:
                raise ValueError("Held object is neither a container or an ingredient")
        return PlayerState(**player_dict)


class OvercookedState(object):
    """A state in OvercookedGridworld."""
    def __init__(self, players, objects, bonus_orders=[], all_orders=[], timestep=0, **kwargs):
        """
        players (list(PlayerState)): Currently active PlayerStates (index corresponds to number)
        objects (dict({tuple:list(FoodState or ContainerState)})):  Dictionary mapping positions (x, y) to FoodState or ContainerState. 
            NOTE: Does NOT include objects held by players (they are in 
            the PlayerState objects).
        bonus_orders (list(dict)):   Current orders worth a bonus
        all_orders (list(dict)):     Current orders allowed at all
        timestep (int):  The current timestep of the state

        """
        bonus_orders = [Recipe.from_dict(order) for order in bonus_orders]
        all_orders = [Recipe.from_dict(order) for order in all_orders]
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        self._bonus_orders = bonus_orders
        self._all_orders = all_orders
        self.timestep = timestep

        assert len(set(self.bonus_orders)) == len(self.bonus_orders), "Bonus orders must not have duplicates"
        assert len(set(self.all_orders)) == len(self.all_orders), "All orders must not have duplicates"
        assert set(self.bonus_orders).issubset(set(self.all_orders)), "Bonus orders must be a subset of all orders"

    @property
    def player_positions(self):
        return tuple([player.position for player in self.players])

    @property
    def player_orientations(self):
        return tuple([player.orientation for player in self.players])

    @property
    def players_pos_and_or(self):
        """Returns a ((pos1, or1), (pos2, or2)) tuple"""
        return tuple(zip(*[self.player_positions, self.player_orientations]))

    @property
    def unowned_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, NOT including
        ones held by players.
        """
        objects_by_type = defaultdict(list)
        for _pos, obj in self.objects.items():
            objects_by_type[obj.name].append(obj)
        return objects_by_type

    @property
    def player_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects held by players.
        """
        player_objects = defaultdict(list)
        for player in self.players:
            if player.has_object():
                player_obj = player.get_object()
                player_objects[player_obj.name].append(player_obj)
        return player_objects

    @property
    def all_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, including
        ones held by players.
        """
        all_objs_by_type = self.unowned_objects_by_type.copy()
        for obj_type, player_objs in self.player_objects_by_type.items():
            all_objs_by_type[obj_type].extend(player_objs)
        return all_objs_by_type

    @property
    def all_objects_list(self):
        all_objects_lists = list(self.all_objects_by_type.values()) + [[], []]
        return reduce(lambda x, y: x + y, all_objects_lists)

    @property
    def all_orders(self):
        return sorted(self._all_orders) if self._all_orders else sorted(Recipe.ALL_RECIPES)

    @property
    def bonus_orders(self):
        return sorted(self._bonus_orders)

    def has_object(self, pos):
        return pos in self.objects

    def get_object(self, pos):
        assert self.has_object(pos)
        return self.objects[pos]

    def add_object(self, obj, pos=None):
        if pos is None:
            pos = obj.position

        assert not self.has_object(pos)
        obj.position = pos
        self.objects[pos] = obj

    def remove_object(self, pos):
        assert self.has_object(pos)
        obj = self.objects[pos]
        del self.objects[pos]
        return obj

    @classmethod
    def from_players_pos_and_or(cls, players_pos_and_or, bonus_orders=[], all_orders=[]):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return cls(
            [PlayerState(*player_pos_and_or) for player_pos_and_or in players_pos_and_or], 
            objects={}, bonus_orders=bonus_orders, all_orders=all_orders)

    @classmethod
    def from_player_positions(cls, player_positions, bonus_orders=[], all_orders=[]):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return cls.from_players_pos_and_or(dummy_pos_and_or, bonus_orders, all_orders)

    def deepcopy(self):
        return OvercookedState(
            players=[player.deepcopy() for player in self.players],
            objects={pos:obj.deepcopy() for pos, obj in self.objects.items()}, 
            bonus_orders=[order.to_dict() for order in self.bonus_orders],
            all_orders=[order.to_dict() for order in self.all_orders],
            timestep=self.timestep)

    def time_independent_equal(self, other):
        order_lists_equal = self.all_orders == other.all_orders and self.bonus_orders == other.bonus_orders

        return isinstance(other, OvercookedState) and \
            self.players == other.players and \
            set(self.objects.items()) == set(other.objects.items()) and \
            order_lists_equal

    def __eq__(self, other):
        return self.time_independent_equal(other) and self.timestep == other.timestep

    def __hash__(self):
        order_list_hash = hash(tuple(self.bonus_orders)) + hash(tuple(self.all_orders))
        return hash(
            (self.players, tuple(self.objects.values()), order_list_hash)
        )

    def __str__(self):
        return 'Players: {}, Objects: {}, Bonus orders: {} All orders: {} Timestep: {}'.format( 
            str(self.players), str(list(self.objects.values())), str(self.bonus_orders), str(self.all_orders), str(self.timestep))

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "bonus_orders": [order.to_dict() for order in self.bonus_orders],
            "all_orders" : [order.to_dict() for order in self.all_orders],
            "timestep" : self.timestep
        }

    @staticmethod
    def from_dict(state_dict):
        state_dict = copy.deepcopy(state_dict)
        state_dict["players"] = [PlayerState.from_dict(p) for p in state_dict["players"]]
        object_list = [ContainerState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = { ob.position : ob for ob in object_list }
        return OvercookedState(**state_dict)


class OvercookedGridworld(object):
    """
    An MDP grid world based off of the Overcooked game.
    TODO: clean the organization of this class further.
    """


    #########################
    # INSTANTIATION METHODS #
    #########################

    def __init__(self, terrain, start_player_positions, start_bonus_orders=[], rew_shaping_params=None, layout_name="unnamed_layout", start_all_orders=[], order_bonus=2, start_state=None, **kwargs):
        """
        terrain: a matrix of strings that encode the MDP layout
        layout_name: string identifier of the layout
        start_player_positions: tuple of positions for both players' starting positions
        start_bonus_orders: List of recipes dicts that are worth a bonus 
        rew_shaping_params: reward given for completion of specific subgoals
        all_orders: List of all available foods the players can make, defaults to all possible recipes if empy list provided
        order_bonus: Multiplicative factor for serving a bonus recipe
        start_state: Default start state returned by get_standard_start_state
        """
        self._configure_recipes(start_all_orders, **kwargs)
        # JYP: need to change `start_all_orders` in case it's not given. WAS dict but now should be food names?
        self.start_all_orders = [r.to_dict() for r in Recipe.ALL_RECIPES] if not start_all_orders else start_all_orders
        self.height = len(terrain)
        self.width = len(terrain[0])
        self.shape = (self.width, self.height)
        self.terrain_mtx = terrain
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.start_player_positions = start_player_positions
        self.num_players = len(start_player_positions)
        self.start_bonus_orders = start_bonus_orders
        self.reward_shaping_params = BASE_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        self.layout_name = layout_name
        self.order_bonus = order_bonus
        self.start_state = start_state
        self._opt_recipe_discount_cache = {}
        self._opt_recipe_cache = {}
        self._prev_potential_params = {}

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params['grid']
        del base_layout_params['grid']
        base_layout_params['layout_name'] = layout_name
        if 'start_state' in base_layout_params:
            # JYP check if it ever goes in here
            assert False
            base_layout_params['start_state'] = OvercookedState.from_dict(base_layout_params['start_state'])

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return OvercookedGridworld.from_grid(grid, base_layout_params, params_to_overwrite)

    @staticmethod
    def from_grid(layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False):
        """
        Returns instance of OvercookedGridworld with terrain and starting 
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = copy.deepcopy(base_layout_params)

        layout_grid = [[c for c in row] for row in layout_grid]
        OvercookedGridworld._assert_valid_grid(layout_grid)

        if "layout_name" not in mdp_config:
            layout_name = "|".join(["".join(line) for line in layout_grid])
            mdp_config["layout_name"] = layout_name

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    layout_grid[y][x] = ' '

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert player_positions[int(c) - 1] is None, 'Duplicate player in grid'
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config.get(k, None)
            if debug:
                print("Overwriting mdp layout standard config value {}:{} -> {}".format(k, curr_val, v))
            mdp_config[k] = v

        return OvercookedGridworld(**mdp_config)

    def _configure_recipes(self, start_all_orders, **kwargs):
        self.recipe_config = {"all_orders" : start_all_orders, **kwargs}
        Recipe.configure(self.recipe_config)

    #####################
    # BASIC CLASS UTILS #
    #####################

    def __eq__(self, other):
        return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
                self.start_player_positions == other.start_player_positions and \
                self.start_bonus_orders == other.start_bonus_orders and \
                self.start_all_orders == other.start_all_orders and \
                self.reward_shaping_params == other.reward_shaping_params and \
                self.layout_name == other.layout_name
    
    def copy(self):
        return OvercookedGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_bonus_orders=self.start_bonus_orders,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name,
            start_all_orders=self.start_all_orders
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_bonus_orders": self.start_bonus_orders,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params),
            "start_all_orders" : self.start_all_orders
        }


    ##############
    # GAME LOGIC #
    ##############

    def get_actions(self, state):
        """
        Returns the list of lists of valid actions for 'state'.

        The ith element of the list is the list of valid actions that player i
        can take.
        """
        self._check_valid_state(state)
        return [self._get_player_actions(state, i) for i in range(len(state.players))]

    def _get_player_actions(self, state, player_num):
        """All actions are allowed to all players in all states."""
        return Action.ALL_ACTIONS

    def _check_action(self, state, joint_action):
        for p_action, p_legal_actions in zip(joint_action, self.get_actions(state)):
            if p_action not in p_legal_actions:
                raise ValueError('Invalid action')

    def add_default_objects(self, state):
        for def_loc, def_obj in CFG_DEFAULT_OBJECTS.items():
            for def_pos in self.get_terrain_locations(def_loc):
                if def_obj in CFG_ALL_CONTAINERS:
                    state.add_object(ContainerState(def_obj, def_pos))
                elif def_obj in CFG_ALL_RAWFOOD:
                    state.add_object(FoodState(def_obj, def_pos))
                else:
                    raise ValueError('Invalid default object')

    def get_standard_start_state(self):
        if self.start_state:
            return self.start_state
        start_state = OvercookedState.from_player_positions(
            self.start_player_positions, bonus_orders=self.start_bonus_orders, all_orders=self.start_all_orders
        )
        self.add_default_objects(start_state)
        return start_state

    def get_random_start_state_fn(self, random_start_pos=False, rnd_obj_prob_thresh=0.0):
        def start_state_fn():
            if random_start_pos:
                valid_positions = self.get_valid_joint_player_positions()
                start_pos = valid_positions[np.random.choice(len(valid_positions))]
            else:
                start_pos = self.start_player_positions

            start_state = OvercookedState.from_player_positions(start_pos, bonus_orders=self.start_bonus_orders, all_orders=self.start_all_orders)
            self.add_default_objects(start_state)

            if rnd_obj_prob_thresh == 0:
                return start_state

            # Arbitrary hard-coding for randomization of objects
            # For each pot, add a random amount of each ingredient with prob rnd_obj_prob_thresh
            # Begin the container cooking with probability rnd_obj_prob_thresh
            empty_containers = self.get_container_states(start_state)["empty"]
            for pot_loc in empty_containers:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    container = start_state.get_object(pot_loc)
                    ingredient_count = {}
                    remaining_count = container.max_ingredients
                    for idx, elem in enumerate(CFG_ALL_INGREDIENTS):
                        if elem in CFG_CONTAINER_INFO[container.name].get('can_add', []):
                            rand_count = int(np.random.randint(low=int(idx==0), high=remaining_count+1))
                            ingredient_count[elem] = rand_count
                            remaining_count -= rand_count

                    q = np.random.rand()
                    cooking_tick = 0 if q < rnd_obj_prob_thresh else -1
                    start_state.objects[pot_loc] = ContainerState.get_container(container.name, pot_loc, ingredient_count, cooking_tick=cooking_tick)

            # For each player, add a random object with prob rnd_obj_prob_thresh
            for player in start_state.players:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    # Different objects have different probabilities
                    obj = np.random.choice(CFG_ALL_OBJECTS)
                    if obj in CFG_ALL_CONTAINERS:
                        ingredient_count = {}
                        remaining_count = CFG_CONTAINER_INFO[obj]['max_ingredients']
                        for idx, elem in enumerate(CFG_ALL_INGREDIENTS):
                            if elem in CFG_CONTAINER_INFO[obj].get('can_add', []):
                                rand_count = int(np.random.randint(low=int(idx==0), high=remaining_count+1))
                                ingredient_count[elem] = rand_count
                                remaining_count -= rand_count
                                if remaining_count <= 0:
                                    break
                        player.set_object(ContainerState.get_container(obj, player.position, ingredient_count, finished=True))
                    else:
                        player.set_object(FoodState(obj, player.position))
            return start_state
        return start_state_fn

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        return False

    def get_state_transition(self, state, joint_action, display_phi=False, motion_planner=None):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered, 
        shaped reward is given only for completion of subgoals 
        (not soup deliveries).
        """
        events_infos = { event : [False] * self.num_players for event in EVENT_TYPES }
        assert not self.is_terminal(state), "Trying to find successor of a terminal state: {}".format(state)
        for action, action_set in zip(joint_action, self.get_actions(state)):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))
        
        new_state = state.deepcopy()

        # Resolve interacts first
        sparse_reward_by_agent, shaped_reward_by_agent = self.resolve_interacts(new_state, joint_action, events_infos)

        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations
        
        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        self.step_environment_effects(new_state)

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)
        infos = {
            "event_infos": events_infos,
            "sparse_reward_by_agent": sparse_reward_by_agent,
            "shaped_reward_by_agent": shaped_reward_by_agent,
        }
        if display_phi:
            assert motion_planner is not None, "motion planner must be defined if display_phi is true"
            infos["phi_s"] = self.potential_function(state, motion_planner)
            infos["phi_s_prime"] = self.potential_function(new_state, motion_planner)
        return new_state, infos

    def resolve_interacts(self, new_state, joint_action, events_infos):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact 
        first and then player 2's, without doing anything like collision checking.
        """
        # JYP: remove this as nothing uses the pot_states except logging functions
        # pot_states = self.get_pot_states(new_state)
        # We divide reward by agent to keep track of who contributed
        sparse_reward, shaped_reward = [0] * self.num_players, [0] * self.num_players 

        for player_idx, (player, action) in enumerate(zip(new_state.players, joint_action)):

            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            # NOTE: we always log pickup/drop before performing it, as that's
            # what the logic of determining whether the pickup/drop is useful assumes

            # If player is holding an object
            if player.has_object():
                player_object = player.get_object()
                # If object in front of player, interact with the object there no matter what the platform underneath
                if new_state.has_object(i_pos):
                    front_object = new_state.get_object(i_pos)
                    # player: container, front: ingredient
                    mydebug(f"*player: {player_object}, front: {front_object}")
                    if type(player_object) is ContainerState and type(front_object) is FoodState:
                        mydebug(f"player: container, front: ingredient")
                        if player_object.can_add(front_object.name):
                            old_container = player_object.deepcopy()
                            # Perform
                            obj = new_state.remove_object(i_pos)
                            player_object.add_ingredient(obj)
                            # Reward
                            shaped_reward[player_idx] += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
                            # Log
                            # self.log_food_to_container(events_infos, new_state, old_container, front_object, obj.name, player_idx)
                            # if obj.name in CFG_ALL_INGREDIENTS:
                            #     events_infos[f'potting_{obj.name}'][player_idx] = True
                            # JYP: may need to change here to first check if container can be place in the place
                            obj = player.remove_object()
                            new_state.add_object(obj, i_pos)
                    # player: ingredient, front: container
                    elif type(player_object) is FoodState and type(front_object) is ContainerState:
                        mydebug(f"player: ingredient, front: container")
                        if front_object.can_add(player_object.name):
                            old_container = front_object.deepcopy()
                            # Perform
                            obj = player.remove_object()
                            front_object.add_ingredient(obj)
                            # Reward
                            shaped_reward[player_idx] += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
                            # Log
                            # self.log_food_to_container(events_infos, new_state, old_container, front_object, obj.name, player_idx)
                            # if obj.name in CFG_ALL_INGREDIENTS:
                            #     events_infos[f'potting_{obj.name}'][player_idx] = True
                    # player: container, front: container
                    elif type(player_object) is ContainerState and type(front_object) is ContainerState:
                        if player_object.is_empty and not front_object.is_empty and front_object.is_ready:
                            mydebug(f"player: empty, front: full")
                            if player_object.can_add(front_object.result_food_name):
                                result_food = front_object.get_result_food()
                                player_object.add_ingredient(result_food)
                        elif front_object.is_empty and not player_object.is_empty and player_object.is_ready:
                            mydebug(f"player: full, front: empty")
                            if front_object.can_add(player_object.result_food_name):
                                result_food = player_object.get_result_food()
                                front_object.add_ingredient(result_food)
                # If no object in front of player, interact with the platform
                else:
                    if CFG_SYMBOL_TO_TERRAIN[terrain_type] == 'deliver':
                        if player_object.name in CFG_DELIVERABLE_CONTAINER and player_object.is_ready:
                            mydebug(f"devliver food")
                            # Perform
                            delivery_rew = self.deliver_food(new_state, player, player_object.get_result_food())
                            # Reward
                            sparse_reward[player_idx] += delivery_rew
                            # Log
                            # events_infos['deliver'][player_idx] = True
                    elif CFG_SYMBOL_TO_TERRAIN[terrain_type] != 'floor':
                        obj_name = player_object.name
                        mydebug(f"place {player_object}")
                        # Log
                        # self.log_object_drop(events_infos, new_state, obj_name, pot_states, player_idx)
                        # Perform
                        new_state.add_object(player.remove_object(), i_pos)
            # If player is not holding an object
            else:
                # If object in front of player, pick up the object
                if new_state.has_object(i_pos):
                    # Log
                    obj_name = new_state.get_object(i_pos).name
                    # self.log_object_pick(events_infos, new_state, obj_name, pot_states, player_idx)
                    # Perform
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)
                    mydebug(f"pick up {obj}")
                # If no object in front of player, interact with the platform
                else:
                    # Pick up an ingredient from the box
                    if CFG_SYMBOL_TO_TERRAIN[terrain_type] in CFG_ALL_RAWFOOD:
                        mydebug(f"pick up {CFG_SYMBOL_TO_TERRAIN[terrain_type]}")
                        # Log
                        # self.log_object_pick(events_infos, new_state, CFG_SYMBOL_TO_TERRAIN[terrain_type], pot_states, player_idx)
                        # Perform
                        new_ing = FoodState(CFG_SYMBOL_TO_TERRAIN[terrain_type], pos)
                        player.set_object(new_ing)
                    # Pick up a dish from the box
                    elif CFG_SYMBOL_TO_TERRAIN[terrain_type] == 'dish':
                        mydebug(f"pick up plate")
                        # Reward
                        # JYP: give this reward all the time instead of checking the if statement
                        # if self.is_dish_pick_useful(new_state, pot_states):
                        shaped_reward[player_idx] += self.reward_shaping_params["DISH_PICKUP_REWARD"]
                        # Log
                        # self.log_object_pick(events_infos, new_state, "dish", pot_states, player_idx)
                        # Perform
                        # JYP : better to remove the cooking_tick and do it some other way.
                        new_cont = ContainerState('dish', pos)
                        player.set_object(new_cont)

        return sparse_reward, shaped_reward

    def get_recipe_value(self, state, recipe, discounted=False, base_recipe=None, potential_params={}):
        """
        Return the reward the player should receive for delivering this recipe

        The player receives 0 if recipe not in all_orders, receives base value * order_bonus
        if recipe is in bonus orders, and receives base value otherwise
        """
        if not discounted:
            if not recipe in state.all_orders:
                return 0
            
            if not recipe in state.bonus_orders:
                return recipe.value

            return self.order_bonus * recipe.value
        else:
            # Calculate missing ingredients needed to complete recipe
            missing_ingredients = list(recipe.ingredients)
            prev_ingredients = list(base_recipe.ingredients) if base_recipe else []
            for ingredient in prev_ingredients:
                missing_ingredients.remove(ingredient)

            value = potential_params['gamma']**recipe.time * self.get_recipe_value(state, recipe, discounted=False)

            for elem in CFG_ALL_INGREDIENTS:
                value *= gamma**(potential_params[f'pot_{elem}_steps'] * missing_ingredients.count(elem))

            return value

    def deliver_food(self, state, player, food):
        """
        Deliver food, and get reward if there is no order list
        or if the type of the delivered food matches the next order.
        """
        # JYP check if this assert isn't needed
        # assert food.name == 'soup', "Tried to deliver something that wasn't soup"
        assert food.is_ready, "Tried to deliever soup that isn't ready"
        player.remove_object()

        return self.get_recipe_value(state, food.recipe)

    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        new_positions, new_orientations = self.compute_new_positions_and_orientations(state.players, joint_action)
        for player_state, new_pos, new_o in zip(state.players, new_positions, new_orientations):
            player_state.update_pos_and_or(new_pos, new_o)

    def compute_new_positions_and_orientations(self, old_player_states, joint_action):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(zip(*[
            self._move_if_direction(p.position, p.orientation, a) \
            for p, a in zip(old_player_states, joint_action)]))
        old_positions = tuple(p.position for p in old_player_states)
        new_positions = self._handle_collisions(old_positions, new_positions)
        return new_positions, new_orientations

    def is_transition_collision(self, old_positions, new_positions):
        # Checking for any players ending in same square
        if self.is_joint_position_collision(new_positions):
            return True
        # Check if any two players crossed paths
        for idx0, idx1 in itertools.combinations(range(self.num_players), 2):
            p1_old, p2_old = old_positions[idx0], old_positions[idx1]
            p1_new, p2_new = new_positions[idx0], new_positions[idx1]
            if p1_new == p2_old and p1_old == p2_new:
                return True
        return False

    def is_joint_position_collision(self, joint_position):
        return any(pos0 == pos1 for pos0, pos1 in itertools.combinations(joint_position, 2))
            
    def step_environment_effects(self, state):
        state.timestep += 1
        for obj in state.objects.values():
            if obj.name in CFG_ALL_CONTAINERS and obj.is_cooking:
                terrain_type = self.get_terrain_type_at_pos(obj.position)
                if CFG_CONTAINER_INFO[obj.name].get('activate_on', None) == CFG_SYMBOL_TO_TERRAIN[terrain_type]:
                    obj.cook()

    def _handle_collisions(self, old_positions, new_positions):
        """If agents collide, they stay at their old locations"""
        if self.is_transition_collision(old_positions, new_positions):
            return old_positions
        return new_positions

    def _get_terrain_type_pos_dict(self):
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict

    def _move_if_direction(self, position, orientation, action):
        """Returns position and orientation that would 
        be obtained after executing action"""
        if action not in Action.MOTION_ACTIONS:
            return position, orientation
        new_pos = Action.move_in_direction(position, action)
        new_orientation = orientation if action == Action.STAY else action
        if new_pos not in self.get_valid_player_positions():
            return position, new_orientation
        return new_pos, new_orientation


    #######################
    # LAYOUT / STATE INFO #
    #######################

    def get_valid_player_positions(self):
        return self.terrain_pos_dict[' ']

    def get_valid_joint_player_positions(self):
        """Returns all valid tuples of the form (p0_pos, p1_pos, p2_pos, ...)"""
        valid_positions = self.get_valid_player_positions() 
        all_joint_positions = list(itertools.product(valid_positions, repeat=self.num_players))
        valid_joint_positions = [j_pos for j_pos in all_joint_positions if not self.is_joint_position_collision(j_pos)]
        return valid_joint_positions

    def get_valid_player_positions_and_orientations(self):
        valid_states = []
        for pos in self.get_valid_player_positions():
            valid_states.extend([(pos, d) for d in Direction.ALL_DIRECTIONS])
        return valid_states

    def get_valid_joint_player_positions_and_orientations(self):
        """All joint player position and orientation pairs that are not
        overlapping and on empty terrain."""
        valid_player_states = self.get_valid_player_positions_and_orientations()

        valid_joint_player_states = []
        for players_pos_and_orientations in itertools.product(valid_player_states, repeat=self.num_players):
            joint_position = [plyer_pos_and_or[0] for plyer_pos_and_or in players_pos_and_orientations]
            if not self.is_joint_position_collision(joint_position):
                valid_joint_player_states.append(players_pos_and_orientations)

        return valid_joint_player_states

    def get_adjacent_features(self, player):
        adj_feats = []
        pos = player.position
        for d in Direction.ALL_DIRECTIONS:
            adj_pos = Action.move_in_direction(pos, d)
            adj_feats.append((adj_pos, self.get_terrain_type_at_pos(adj_pos)))
        return adj_feats

    def get_terrain_type_at_pos(self, pos):
        x, y = pos
        return self.terrain_mtx[y][x]

    def get_terrain_locations(self, terrain_type):
        return list(self.terrain_pos_dict[CFG_TERRAIN_TO_SYMBOL[terrain_type]])

    @property
    def num_containers(self):
        return len(self.get_terrain_locations('stove'))

    def get_container_states(self, state):
        """Returns dict with structure:
        {
        empty: [positions of empty containers]
        'x_items': [containers with x items that have yet to start cooking],
        'cooking': [containers that are cooking but not ready]
        'ready': [ready containers],
        }
        NOTE: all returned containers are just positions
        """
        container_states_dict = defaultdict(lambda: defaultdict(list))
        all_objects = state.all_objects_by_type()
        all_containers = [obj for cont in CFG_ALL_CONTAINERS for obj in all_objects[cont]]

        for container in all_containers:
            if container.is_empty:
                container_states_dict['empty'].append(container.position)
            elif container.is_cooking:
                container_states_dict['cooking'].append(container.position)
            elif container.is_ready:
                container_states_dict['ready'].append(container.position)
            else:
                num_ingredients = len(container.ingredients)
                container_states_dict[f"{num_ingredients}_items"].append(container.position)

        return container_states_dict

    def old_get_container_states(self, state):
        """Returns dict with structure:
        {container_type:
            {
            empty: [positions of empty containers]
            'x_items': [containers with x items that have yet to start cooking],
            'cooking': [containers that are cooking but not ready]
            'ready': [ready containers],
            }
        }
        NOTE: all returned containers are just positions
        """
        container_states_dict = defaultdict(lambda: defaultdict(list))
        all_objects = state.all_objects_by_type()
        all_containers = [obj for cont in CFG_ALL_CONTAINERS for obj in all_objects[cont]]

        for container in all_containers:
            if container.is_empty:
                container_states_dict[container.name]['empty'].append(container.position)
            elif container.is_cooking:
                container_states_dict[container.name]['cooking'].append(container.position)
            elif container.is_ready:
                container_states_dict[container.name]['ready'].append(container.position)
            else:
                num_ingredients = len(container.ingredients)
                container_states_dict[container.name][f"{num_ingredients}_items"].append(container.position)

        return container_states_dict

    def get_counter_objects_dict(self, state, counter_subset=None):
        """Returns a dictionary of pos:objects on counters by type"""
        if counter_subset is None:
            counter_subset = self.terrain_pos_dict['T'] + self.terrain_pos_dict['O'] + \
                             self.terrain_pos_dict['D'] + self.terrain_pos_dict['S'] + self.terrain_pos_dict['X']
        counter_objects_dict = defaultdict(list)
        for obj in state.objects.values():
            if obj.position in counters_considered:
                counter_objects_dict[obj.name].append(obj.position)
        return counter_objects_dict

    def get_empty_counter_locations(self, state):
        counter_locations = self.get_terrain_locations('tomato') + self.get_terrain_locations('onion') + \
                            self.get_terrain_locations('dish') + self.get_terrain_locations('stove') + self.get_terrain_locations('counter')
        return [pos for pos in counter_locations if not state.has_object(pos)]

    def get_empty_containers(self, container_states):
        """Returns containers that have 0 items in them"""
        return container_states["empty"]

    def get_non_empty_containers(self, container_states):
        return self.get_full_containers(container_states) + self.get_partially_full_containers(container_states)

    def get_ready_containers(self, container_states):
        return container_states['ready']

    def get_cooking_containers(self, container_states):
        return container_states['cooking']

    def get_full_but_not_cooking_containers(self, container_states):
        return container_states['{}_items'.format(CFG_MAX_NUM_INGREDIENTS)]

    def get_full_containers(self, container_states):
        return self.get_cooking_containers(container_states) + self.get_ready_containers(container_states) + self.get_full_but_not_cooking_containers(container_states)

    def get_partially_full_containers(self, container_states):
        return list(set().union(*[container_states['{}_items'.format(i)] for i in range(1, CFG_MAX_NUM_INGREDIENTS)]))

    def _check_valid_state(self, state):
        """Checks that the state is valid.

        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)
        """
        all_objects = list(state.objects.values())
        for player_state in state.players:
            # Check that players are not on terrain
            pos = player_state.position
            assert pos in self.get_valid_player_positions()

            # Check that held objects have the same position
            if player_state.held_object is not None:
                all_objects.append(player_state.held_object)
                assert player_state.held_object.position == player_state.position

        for obj_pos, obj_state in state.objects.items():
            # Check that the hash key position agrees with the position stored
            # in the object state
            assert obj_state.position == obj_pos
            # Check that non-held objects are on terrain
            assert self.get_terrain_type_at_pos(obj_pos) != ' '

        # Check that players and non-held objects don't overlap
        all_pos = [player_state.position for player_state in state.players]
        all_pos += [obj_state.position for obj_state in state.objects.values()]
        assert len(all_pos) == len(set(all_pos)), "Overlapping players or objects"

        # Check that objects have a valid state
        for obj_state in all_objects:
            assert obj_state.is_valid()

    def find_free_counters_valid_for_both_players(self, state, mlam):
        """Finds all empty counter locations that are accessible to both players"""
        one_player, other_player = state.players
        free_counters = self.get_empty_counter_locations(state)
        free_counters_valid_for_both = []
        for free_counter in free_counters:
            goals = mlam.motion_planner.motion_goals_for_pos[free_counter]
            if any([mlam.motion_planner.is_valid_motion_start_goal_pair(one_player.pos_and_or, goal) for goal in goals]) and \
            any([mlam.motion_planner.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal) for goal in goals]):
                free_counters_valid_for_both.append(free_counter)
        return free_counters_valid_for_both

    def _get_optimal_possible_recipe(self, state, recipe, discounted, potential_params, return_value):
        """
        Traverse the recipe-space graph using DFS to find the best possible recipe that can be made
        from the current recipe

        Because we can't have empty recipes, we handle the case by letting recipe==None be a stand-in for empty recipe
        """
        start_recipe = recipe
        visited = set()
        stack = []
        best_recipe = recipe
        best_value = 0
        if not recipe:
            for ingredient in CFG_ALL_INGREDIENTS:
                stack.append(Recipe([ingredient]))
        else:
            stack.append(recipe)

        while stack:
            curr_recipe = stack.pop()
            if curr_recipe not in visited:
                visited.add(curr_recipe)
                curr_value = self.get_recipe_value(state, curr_recipe, base_recipe=start_recipe, discounted=discounted, potential_params=potential_params)
                if curr_value > best_value:
                    best_value, best_recipe = curr_value, curr_recipe
                
                for neighbor in curr_recipe.neighbors():
                    if not neighbor in visited:
                        stack.append(neighbor)
        
        if return_value:
            return best_recipe, best_value
        return best_recipe


    def get_optimal_possible_recipe(self, state, recipe, discounted=False, potential_params={}, return_value=False):
        """
        Return the best possible recipe that can be made starting with the given `recipe`
        Uses self._optimal_possible_recipe as a cache to avoid re-computing. This only works because
        the recipe values are currently static (i.e. bonus_orders doesn't change). Would need to have cache
        flushed if order dynamics are introduced
        """
        # ingredients = tuple(sorted(ingredients)) if ingredients else None
        cache_valid = not discounted or self._prev_potential_params == potential_params
        if not cache_valid:
            if discounted:
                self._opt_recipe_discount_cache = {}
            else:
                self._opt_recipe_cache = {}

        if discounted:
            cache = self._opt_recipe_discount_cache
            self._prev_potential_params = potential_params
        else:
            cache = self._opt_recipe_cache

        if recipe not in cache:
            # Compute best recipe now and store in cache for later use
            opt_recipe, value = self._get_optimal_possible_recipe(state, recipe, discounted=discounted, potential_params=potential_params, return_value=True)
            cache[recipe] = (opt_recipe, value)

        # Return best recipe (and value) from cache
        if return_value:
            return cache[recipe]
        return cache[recipe][0]
        



    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'P' (pot), 'D' (dish supply), '_' (serving location), '1' 
        (player 1), '2' (player 2), '{I}' (ingredient supply),
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must not be free spaces
        def is_not_free(c):
            return c in CFG_TERRAIN_TO_SYMBOL.values()

        for y in range(height):
            assert is_not_free(grid[y][0]), 'Left border must not be free'
            assert is_not_free(grid[y][-1]), 'Right border must not be free'
        for x in range(width):
            assert is_not_free(grid[0][x]), 'Top border must not be free'
            assert is_not_free(grid[-1][x]), 'Bottom border must not be free'

        all_elements = [element for row in grid for element in row]
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(range(1, num_players + 1)), "Some players were missing"

        assert all(c in '123456789' + "".join(CFG_TERRAIN_TO_SYMBOL.values()) for c in all_elements), 'Invalid character in grid'
        assert all_elements.count('1') == 1, "'1' must be present exactly once"
        assert all_elements.count('D') >= 1, "'D' must be present at least once"
        assert all_elements.count('_') >= 1, "'_' must be present at least once"
        # JYP uncomment this?
        # assert all_elements.count('P') >= 1, "'P' must be present at least once"
        # assert all_elements.count('S') >= 1, "'S' must be present at least once"
        # assert sum([all_elements.count(CFG_TERRAIN_TO_SYMBOL[elem]) for elem in CFG_ALL_RAWFOOD]) >= 1, f"Some ingredient must be present at least once"


    ################################
    # EVENT LOGGING HELPER METHODS #
    ################################

    # def log_food_to_container(self, events_infos, state, old_cont, new_cont, food_name, player_index):
    #     """Player added an ingredient to a container"""
    #     food_add_key = f"{food_name}_to_{new_cont.name}"
    #     if food_add_key not in events_infos:
    #         raise ValueError("Unknown event {}".format(food_add_key))
    #     events_infos[food_add_key][player_index] = True

    #     POTTING_FNS = {
    #         "optimal" : self.is_potting_optimal,
    #         "catastrophic" : self.is_potting_catastrophic,
    #         "viable" : self.is_potting_viable,
    #         "useless" : self.is_potting_useless
    #     }

    #     for outcome, outcome_fn in POTTING_FNS.items():
    #         if outcome_fn(state, old_cont, new_cont):
    #             potting_key = f"{outcome}_{food_add_key}"
    #             events_infos[potting_key][player_index] = True

    
    # def log_object_pick(self, events_infos, state, obj_name, pot_states, player_index):
    #     """Player picked an object up from a counter or a dispenser"""
    #     obj_pick_key = f"{obj_name}_pick"
    #     if obj_pick_key not in events_infos:
    #         raise ValueError("Unknown event {}".format(obj_pick_key))
    #     events_infos[obj_pick_key][player_index] = True
        
    #     USEFUL_PICKUP_FNS = {elem : self.is_ingredient_pick_useful for elem in CFG_ALL_RAWFOOD}
    #     USEFUL_PICKUP_FNS["dish"] = self.is_dish_pick_useful
    #     if obj_name in USEFUL_PICKUP_FNS:
    #         if USEFUL_PICKUP_FNS[obj_name](state, pot_states, player_index):
    #             obj_useful_key = f"useful_{obj_pick_key}"
    #             events_infos[obj_useful_key][player_index] = True

    # def log_object_drop(self, events_infos, state, obj_name, pot_states, player_index):
    #     """Player dropped the object on a counter"""
    #     obj_drop_key = f"{obj_name}_drop"
    #     if obj_drop_key not in events_infos:
    #         raise ValueError("Unknown event {}".format(obj_drop_key))
    #     events_infos[obj_drop_key][player_index] = True
        
    #     USEFUL_DROP_FNS = {elem : self.is_ingredient_drop_useful for elem in CFG_ALL_RAWFOOD}
    #     USEFUL_DROP_FNS["dish"] = self.is_dish_drop_useful
    #     if obj_name in USEFUL_DROP_FNS:
    #         if USEFUL_DROP_FNS[obj_name](state, pot_states, player_index):
    #             obj_useful_key = f"useful_{obj_drop_key}"
    #             events_infos[obj_useful_key][player_index] = True

    # def is_dish_pick_useful(self, state, pot_states, player_index=None):
    #     """
    #     NOTE: this only works if self.num_players == 2
    #     Useful if:
    #     - Pot is ready/cooking and there is no player with a dish               \ 
    #     - 2 pots are ready/cooking and there is one player with a dish          | -> number of dishes in players hands < number of ready/cooking/partially full soups 
    #     - Partially full pot is ok if the other player is on course to fill it  /

    #     We also want to prevent picking up and dropping dishes, so add the condition
    #     that there must be no dishes on counters
    #     """
    #     if self.num_players != 2: return False

    #     # This next line is to prevent reward hacking (this logic is also used by reward shaping)
    #     dishes_on_counters = self.get_counter_objects_dict(state)["dish"]
    #     no_dishes_on_counters = len(dishes_on_counters) == 0

    #     num_player_dishes = len(state.player_objects_by_type['dish'])
    #     non_empty_pots = len(self.get_ready_containers(pot_states) + self.get_cooking_containers(pot_states) + self.get_partially_full_containers(pot_states))
    #     return no_dishes_on_counters and num_player_dishes < non_empty_pots

    # def is_dish_drop_useful(self, state, pot_states, player_index):
    #     """
    #     NOTE: this only works if self.num_players == 2
    #     Useful if:
    #     - Ingredient is needed (all pots are non-full)
    #     - Nobody is holding ingredients
    #     """
    #     if self.num_players != 2: return False
    #     all_non_full = len(self.get_full_containers(pot_states)) == 0
    #     other_player = state.players[1 - player_index]
    #     other_player_holding_ingredient = other_player.has_object() and other_player.get_object().name in CFG_ALL_RAWFOOD
    #     return all_non_full and not other_player_holding_ingredient

    # def is_ingredient_pick_useful(self, state, pot_states, player_index):
    #     """
    #     NOTE: this only works if self.num_players == 2
    #     Always useful unless:
    #     - All pots are full & other agent is not holding a dish
    #     """
    #     if self.num_players != 2: return False
    #     all_pots_full = self.num_containers == len(self.get_full_containers(pot_states))
    #     other_player = state.players[1 - player_index]
    #     other_player_has_dish = other_player.has_object() and other_player.get_object().name == "dish"
    #     return not (all_pots_full and not other_player_has_dish)
        
    # def is_ingredient_drop_useful(self, state, pot_states, player_index):
    #     """
    #     NOTE: this only works if self.num_players == 2
    #     Useful if:
    #     - Dish is needed (all pots are full)
    #     - Nobody is holding a dish
    #     """
    #     if self.num_players != 2: return False
    #     all_pots_full = len(self.get_full_containers(pot_states)) == self.num_containers
    #     other_player = state.players[1 - player_index]
    #     other_player_holding_dish = other_player.has_object() and other_player.get_object().name == "dish"
    #     return all_pots_full and not other_player_holding_dish

    # def is_potting_optimal(self, state, old_soup, new_soup):
    #     """
    #     True if the highest valued soup possible is the same before and after the potting
    #     """
    #     old_recipe = Recipe(old_soup.ingredients) if old_soup.ingredients else None
    #     new_recipe = Recipe(new_soup.ingredients)
    #     old_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, old_recipe))
    #     new_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, new_recipe))
    #     return old_val == new_val

    # def is_potting_viable(self, state, old_soup, new_soup):
    #     """
    #     True if there exists a non-zero reward soup possible from new ingredients
    #     """
    #     new_recipe = Recipe(new_soup.ingredients)
    #     new_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, new_recipe))
    #     return new_val > 0

    # def is_potting_catastrophic(self, state, old_soup, new_soup):
    #     """
    #     True if no non-zero reward soup is possible from new ingredients
    #     """
    #     old_recipe = Recipe(old_soup.ingredients) if old_soup.ingredients else None
    #     new_recipe = Recipe(new_soup.ingredients)
    #     old_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, old_recipe))
    #     new_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, new_recipe))
    #     return old_val > 0 and new_val == 0

    # def is_potting_useless(self, state, old_soup, new_soup):
    #     """
    #     True if ingredient added to a soup that was already gauranteed to be worth at most 0 points
    #     """
    #     old_recipe = Recipe(old_soup.ingredients) if old_soup.ingredients else None
    #     old_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, old_recipe))
    #     return old_val == 0



    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        grid_string = ""
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, element in enumerate(terrain_row):
                grid_string_add = ""
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                    assert len(player_idx_lst) == 1
                    grid_string_add += Action.ACTION_TO_CHAR[orientation] + str(player_idx_lst[0])

                    player_obj = player.held_object
                    if player_obj:
                        grid_string_add += str(player_obj)
                else:
                    grid_string_add += element
                    if state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string_add += str(state_obj)

                grid_string += grid_string_add
                grid_string += "".join([" "] * (7 - len(grid_string_add)))
                grid_string += " "

            grid_string += "\n\n"
        
        if state.bonus_orders:
            grid_string += "Bonus orders: {}\n".format(
                state.bonus_orders
            )
        # grid_string += "State potential value: {}\n".format(self.potential_function(state))
        return grid_string

    ###################
    # STATE ENCODINGS #
    ###################

    def get_lossless_state_encoding_shape(self):
        return np.array(list(self.shape) + [26])


    def lossless_state_encoding(self, overcooked_state, horizon=400, debug=False):
        """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
        assert self.num_players == 2, "Functionality has to be added to support encondings for > 2 players"
        assert type(debug) is bool
        terrain_location_features = [f"{elem}_terrain_loc" for elem in CFG_ALL_TERRAINS]
        object_location_features = [f"{elem}_object_loc" for elem in CFG_ALL_OBJECTS]
        object_state_features = [f"{ingredient}_in_{container}" for ingredient in CFG_ALL_INGREDIENTS for container in CFG_ALL_CONTAINERS] + \
                                [elem for sublist in [[f"{container}_time_left", f"{container}_done"] for container in CFG_ALL_CONTAINERS] for elem in sublist]
        urgency_features = ["urgency"]
        all_objects = overcooked_state.all_objects_list

        def make_layer(position, value):
                layer = np.zeros(self.shape)
                layer[position] = value
                return layer

        def process_for_player(primary_agent_idx):
            # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
            other_agent_idx = 1 - primary_agent_idx
            ordered_player_features = ["player_{}_loc".format(primary_agent_idx), "player_{}_loc".format(other_agent_idx)] + \
                        ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
                        for i, d in itertools.product([primary_agent_idx, other_agent_idx], Direction.ALL_DIRECTIONS)]

            LAYERS = ordered_player_features + terrain_location_features + object_location_features + object_state_features + urgency_features
            state_mask_dict = {k: np.zeros(self.shape) for k in LAYERS}

            # MAP LAYERS
            if horizon - overcooked_state.timestep < 40:
                state_mask_dict["urgency"] = np.ones(self.shape)

            for elem in CFG_ALL_TERRAINS:
                for loc in self.get_terrain_locations(elem):
                    state_mask_dict[f"{elem}_terrain_loc"][loc] = 1

            # PLAYER LAYERS
            for i, player in enumerate(overcooked_state.players):
                player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
                state_mask_dict["player_{}_loc".format(i)] = make_layer(player.position, 1)
                state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)] = make_layer(player.position, 1)

            # OBJECT & STATE LAYERS
            for obj in all_objects:
                if obj.name in CFG_ALL_CONTAINERS:
                    state_mask_dict[f"{obj.name}_object_loc"] += make_layer(obj.position, 1)
                    state_mask_dict[f"{obj.name}_time_left"] += make_layer(obj.position, obj.cook_time - obj._cooking_tick)
                    state_mask_dict[f"{obj.name}_done"] += make_layer(obj.position, int(obj.is_ready))
                    ingredients_dict = Counter(obj.ingredients)
                    for elem in CFG_ALL_INGREDIENTS:
                        state_mask_dict[f"{elem}_in_{obj.name}"] += make_layer(obj.position, ingredients_dict[elem])
                elif obj.name in CFG_ALL_RAWFOOD:
                    state_mask_dict[f"{obj.name}_object_loc"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized object")

            if debug:
                print("terrain----")
                print(np.array(self.terrain_mtx))
                print("-----------")
                print(len(LAYERS))
                print(len(state_mask_dict))
                for k, v in state_mask_dict.items():
                    print(k)
                    print(np.transpose(v, (1, 0)))

            # Stack of all the state masks, order decided by order of LAYERS
            state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in LAYERS])
            state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
            assert state_mask_stack.shape[:2] == self.shape
            assert state_mask_stack.shape[2] == len(LAYERS)
            # NOTE: currently not including time left or order_list in featurization
            return np.array(state_mask_stack).astype(int)

        # NOTE: Currently not very efficient, a decent amount of computation repeated here
        num_players = len(overcooked_state.players)
        final_obs_for_players = tuple(process_for_player(i) for i in range(num_players))
        return final_obs_for_players

    # JYP: need to change these hardcoded numbers
    def get_featurize_state_shape(self, num_containers=2):
        num_pot_features = 10
        base_features = 28
        total_features = self.num_players * (num_containers * num_pot_features + base_features)
        return (total_features,)

    def featurize_state(self, overcooked_state, mlam, num_containers=2, **kwargs):
        """
        Encode state with some manually designed features. Works for arbitrary number of players

        Arguments:
            overcooked_state (OvercookedState): state we wish to featurize
            mlam (MediumLevelActionManager): to be used for distance computations necessary for our higher-level feature encodings
            num_containers (int): Encode the state (ingredients, whether cooking or not, etc) of the 'num_containers' closest containers to each player. 
                If i < num_containers containers are reachable by player i, then containers [i+1, num_containers] are encoded as all zeros. Changing this 
                impacts the shape of the feature encoding
        
        Returns:
            ordered_features (list[np.Array]): The ith element contains a player-centric featurized view for the ith player

            The encoding for player i is as follows:

                [player_i_features, other_player_features player_i_dist_to_other_players, player_i_position]

                player_{i}_features (length num_containers*10 + 24):
                    pi_orientation: length 4 one-hot-encoding of direction currently facing
                    pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
                    pi_wall_{j}: {0, 1} boolean value of whether player i has wall immediately in direction j
                    pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
                    pi_cloest_soup_n_{onions|tomatoes}: int value for number of this ingredient in closest soup
                    pi_closest_container_{j}_exists: {0, 1} depending on whether jth closest container found. If 0, then all other container features are 0. Note: can
                        be 0 even if there are more than j containers on layout, if the container is not reachable by player i
                    pi_closest_container_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest container
                    pi_closest_container_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest container
                    pi_closest_container_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
                    pi_closest_container_{j}: (dx, dy) to jth closest container from player i location

                other_player_features (length (num_players - 1)*(num_containers*10 + 24)):
                    ordered concatenation of player_{j}_features for j != i
                
                player_i_dist_to_other_players (length (num_players - 1)*2):
                    [player_j.pos - player_i.pos for j != i]

                player_i_position (length 2)
        """

        all_features = {}

        def concat_dicts(a, b):
            return {**a, **b}

        def make_closest_feature(idx, player, name, locations):
            """
            Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict
            """
            feat_dict = {} 
            obj = None
            held_obj = player.held_object
            if held_obj and held_obj.name == name:
                obj = held_obj
                feat_dict[f"p{idx}_closest_{name}"] = (0, 0)
            else:
                loc, deltas = self.get_deltas_to_closest_location(player, locations, mlam)
                if loc and overcooked_state.has_object(loc):
                    obj = overcooked_state.get_object(loc)
                feat_dict[f"p{idx}_closest_{name}"] = deltas

            if name in CFG_ALL_CONTAINERS:
                ingredients_cnt = {}
                if obj:
                    ingredients_cnt = Counter(obj.ingredients)
                for elem in CFG_ALL_INGREDIENTS:
                    feat_dict[f"p{idx}_closest_{name}_n_{elem}"] = [ingredients_cnt[elem] if elem in ingredients_cnt else 0]

            return feat_dict

        def make_container_feature(idx, player, container_idx, container_loc, container_states):
            """
            Encode container at container_loc relative to 'player'
            """
            # Container doesn't exist
            feat_dict = {}
            if not container_loc:
                feat_dict[f"p{idx}_closest_container_{container_idx}_exists"] = [0]
                feat_dict[f"p{idx}_closest_container_{container_idx}_is_empty"] = [0]
                feat_dict[f"p{idx}_closest_container_{container_idx}_is_full"] = [0]
                feat_dict[f"p{idx}_closest_container_{container_idx}_is_cooking"] = [0]
                feat_dict[f"p{idx}_closest_container_{container_idx}_is_ready"] = [0]
                feat_dict[f"p{idx}_closest_container_{container_idx}_cook_time"] = [0]
                for elem in CFG_ALL_INGREDIENTS:
                    feat_dict[f"p{idx}_closest_container_{container_idx}_num_{elem}"] = [0]
                feat_dict[f"p{idx}_closest_container_{container_idx}"] = (0, 0)
                return feat_dict
            
            # Get position information
            deltas = self.get_deltas_to_location(player, container_loc)

            # Get container state info
            is_empty = int(container_loc in self.get_empty_containers(container_states))
            is_full = int(container_loc in self.get_full_containers(container_states))
            is_cooking = int(container_loc in self.get_cooking_containers(container_states))
            is_ready = int(container_loc in self.get_ready_containers(container_states))

            # Get soup state info
            ingredients_cnt = {}
            cook_time_remaining = 0
            if not is_empty:
                soup = overcooked_state.get_object(container_loc)
                ingredients_cnt = Counter(soup.ingredients)
                cook_time_remaining = 0 if soup.is_idle else soup.cook_time_remaining
            
            # Encode container and soup info
            feat_dict[f"p{idx}_closest_container_{container_idx}_exists"] = [1]
            feat_dict[f"p{idx}_closest_container_{container_idx}_is_empty"] = [is_empty]
            feat_dict[f"p{idx}_closest_container_{container_idx}_is_full"] = [is_full]
            feat_dict[f"p{idx}_closest_container_{container_idx}_is_cooking"] = [is_cooking]
            feat_dict[f"p{idx}_closest_container_{container_idx}_is_ready"] = [is_ready]
            for elem in CFG_ALL_INGREDIENTS:
                feat_dict[f"p{idx}_closest_container_{container_idx}_num_{elem}"] = [ingredients_cnt[elem] if elem in ingredients_cnt else 0]
            feat_dict[f"p{idx}_closest_container_{container_idx}_cook_time"] = [cook_time_remaining]
            feat_dict[f"p{idx}_closest_container_{container_idx}"] = deltas

            return feat_dict

            


        IDX_TO_OBJ = CFG_ALL_OBJECTS
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        container_states = self.get_container_states(overcooked_state)

        for i, player in enumerate(overcooked_state.players):
            # Player info
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest feature for each object type
            for elem in CFG_ALL_RAWFOOD:
                all_features = concat_dicts(all_features, make_closest_feature(i, player, elem, self.get_terrain_locations(elem) + counter_objects[elem]))
            all_features = concat_dicts(all_features, make_closest_feature(i, player, "deliver", self.get_terrain_locations("deliver")))
            all_features = concat_dicts(all_features, make_closest_feature(i, player, "empty_counter", self.get_empty_counter_locations(overcooked_state)))

            # Closest container info
            for elem in CFG_ALL_CONTAINERS:
                _, closest_container_loc = mlam.motion_planner.min_cost_to_feature(player.pos_and_or, stove_locations, with_argmin=True)
                pot_features = make_pot_feature(i, player, pot_idx, closest_pot_loc, container_states)
                all_features = concat_dicts(all_features, pot_features)

            pot_locations = self.get_terrain_locations('pot').copy()
            for pot_idx in range(num_containers):
                _, closest_pot_loc = mlam.motion_planner.min_cost_to_feature(player.pos_and_or, stove_locations, with_argmin=True)
                pot_features = make_pot_feature(i, player, pot_idx, closest_pot_loc, container_states)
                all_features = concat_dicts(all_features, pot_features)

                if closest_pot_loc:
                    stove_locations.remove(closest_pot_loc)

            # Adjacent features info
            for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
                _, feat = pos_and_feat
                all_features["p{}_wall_{}".format(i, direction)] = [0] if feat == ' ' else [1]

        # Convert all list and tuple values to np.arrays
        features_np = {k: np.array(v) for k, v in all_features.items()}

        player_features = [] # Non-position player-specific features
        player_absolute_positions = [] # Position player-specific features
        player_relative_positions = [] # Relative position player-specific features

        # Compute all player-centric features for each player
        for i, player_i in enumerate(overcooked_state.players):
            # All absolute player-centric features
            player_i_dict = {k: v for k, v in features_np.items() if k[:2] == "p{}".format(i)}
            features = np.concatenate(list(player_i_dict.values()))
            abs_pos = np.array(player_i.position)

            # Calculate position relative to all other players
            rel_pos = []
            for player_j in overcooked_state.players:
                if player_i == player_j:
                    continue
                pj_rel_to_pi = np.array(pos_distance(player_j.position, player_i.position))
                rel_pos.append(pj_rel_to_pi)
            rel_pos = np.concatenate(rel_pos)

            player_features.append(features)
            player_absolute_positions.append(abs_pos)
            player_relative_positions.append(rel_pos)
        
        # Compute a symmetric, player-centric encoding of features for each player
        ordered_features = []
        for i, player_i in enumerate(overcooked_state.players):
            player_i_features = player_features[i]
            player_i_abs_pos = player_absolute_positions[i]
            player_i_rel_pos = player_relative_positions[i]
            other_player_features = np.concatenate([feats for j, feats in enumerate(player_features) if j != i])
            player_i_ordered_features = np.squeeze(np.concatenate([player_i_features, other_player_features, player_i_rel_pos, player_i_abs_pos]))
            ordered_features.append(player_i_ordered_features)

        return ordered_features


    def get_deltas_to_closest_location(self, player, locations, mlam):
        _, closest_loc = mlam.motion_planner.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
        deltas = self.get_deltas_to_location(player, closest_loc)
        return closest_loc, deltas
        

    def get_deltas_to_location(self, player, location):
        if location is None:
            # "any object that does not exist or I am carrying is going to show up as a (0,0)
            # but I can disambiguate the two possibilities by looking at the features 
            # for what kind of object I'm carrying"
            return (0, 0)
        dy_loc, dx_loc = pos_distance(location, player.position)
        return dy_loc, dx_loc


    ###############################
    # POTENTIAL REWARD SHAPING FN #
    ###############################

    def potential_function(self, state, mp, gamma=0.99):
        """
        Essentially, this is the ɸ(s) function.

        The main goal here to to approximately infer the actions of an optimal agent, and derive an estimate for the value
        function of the optimal policy. The perfect potential function is indeed the value function

        At a high level, we assume each agent acts independetly, and greedily optimally, and then, using the decay factor "gamma", 
        we calculate the expected discounted reward under this policy

        Some implementation details:
            * the process of delivering a soup is broken into 4 steps
                * Step 1: placing the first ingredient into an empty pot
                * Step 2: placing the remaining ingredients in the pot
                * Step 3: cooking the soup/retreiving a dish with which to serve the soup
                * Step 4: delivering the soup once it is in a dish
            * Here is an exhaustive list of the greedy assumptions made at each step
                * step 1:
                    * If an agent is holding an ingredient that could be used to cook an optimal soup, it will use it in that soup
                    * If no such optimal soup exists, but there is an empty pot, the agent will place the ingredient there
                    * If neither of the above cases holds, no potential is awarded for possessing the ingredient
                * step 2:
                    * The agent will always try to cook the highest valued soup possible based on the current ingredients in a pot
                    * Any agent possessing a missing ingredient for an optimal soup will travel directly to the closest such pot
                    * If the optimal soup has all ingredients, the closest agent not holding anything will go to cook it
                * step 3:
                    * Any player holding a dish attempts to serve the highest valued soup based on recipe values and cook time remaining
                * step 4:
                    * Any agent holding a soup will go directly to the nearest serving area
            * At every step, the expected reward is discounted by multiplying the optimal reward by gamma ^ (estimated #steps to complete greedy action)
            * In the case that certain actions are infeasible (i.e. an agent is holding a soup in step 4, but no path exists to a serving
              area), estimated number of steps in order to complete the action defaults to `max_steps`
            * Cooperative behavior between the two agents is not considered for complexity reasons
            * Soups that are worth <1 points are rounded to be worth 1 point. This is to incentivize the agent to cook a worthless soup
              that happens to be in a pot in order to free up the pot

        Parameters:
            state: OvercookedState instance representing the state to evaluate potential for
            mp: MotionPlanner instance used to calculate gridworld distances to objects
            gamma: float, discount factor
            max_steps: int, number of steps a high level action is assumed to take in worst case
        
        Returns
            phi(state), the potential of the state
        """

        # Constants needed for potential function
        potential_params = {
            'gamma' : gamma,
            **CFG_POTENTIAL_CONSTANTS.get(self.layout_name, CFG_POTENTIAL_CONSTANTS['default'])
        }
        for elem in CFG_ALL_INGREDIENTS:
            potential_params[f"{elem}_value"] = Recipe._ingredient_value[elem] if Recipe._ingredient_value[elem] else CFG_DEFAULT_POTENTIAL[elem]
        container_states = self.get_container_states(state)

        # Base potential value is the geometric sum of making optimal foods infinitely
        opt_recipe, discounted_opt_recipe_value = self.get_optimal_possible_recipe(state, None, discounted=True, potential_params=potential_params, return_value=True)
        opt_recipe_value = self.get_recipe_value(state, opt_recipe)
        discount = discounted_opt_recipe_value / opt_recipe_value
        steady_state_value = (discount / (1 - discount)) * opt_recipe_value
        potential = steady_state_value

        # Get list of all containers that have >0 ingredients, sorted based on value of best possible recipe 
        idle_containers = [state.get_object(pos) for pos in self.get_full_but_not_cooking_containers(container_states)]
        idle_containers.extend([state.get_object(pos) for pos in self.get_partially_full_containers(container_states)])
        idle_containers = sorted(idle_containers, key=lambda container : self.get_optimal_possible_recipe(state, Recipe(container.ingredients), discounted=True, potential_params=potential_params, return_value=True)[1], reverse=True)

        # Build mapping of non_idle containers to the potential value each one will contribue
        # Default potential value is maximimal discount for last two steps applied to optimal recipe value
        cooking_containers = [state.get_object(pos) for pos in self.get_cooking_containers(container_states)]
        done_containers = [state.get_object(pos) for pos in self.get_ready_containers(container_states)]
        non_idle_container_vals = { container : gamma**(potential_params['max_delivery_steps'] + max(potential_params['max_pickup_steps'], container.cook_time - container._cooking_tick)) * max(self.get_recipe_value(state, container.recipe), 1) for container in cooking_containers + done_containers }

        # Get descriptive list of players based on different attributes
        # Note that these lists are mutually exclusive
        players_holding_soups = [player for player in state.players if player.has_object() and player.get_object().name == 'soup']
        players_holding_dishes = [player for player in state.players if player.has_object() and player.get_object().name == 'dish']
        players_holding_ingredient = [[player for player in state.players if player.has_object() and player.get_object().name == elem] for elem in CFG_ALL_RAWFOOD]
        players_holding_nothing = [player for player in state.players if not player.has_object()]

        ### Step 4 potential ###

        # Add potential for each player with a soup
        for player in players_holding_soups:
            # Even if delivery_dist is infinite, we still award potential (as an agent might need to pass the soup to other player first)
            delivery_dist = mp.min_cost_to_feature(player.pos_and_or, self.terrain_pos_dict['_'])
            potential += gamma**min(delivery_dist, potential_params['max_delivery_steps']) * max(self.get_recipe_value(state, player.get_object().recipe), 1)



        ### Step 3 potential ###

        # Reweight each non-idle soup value based on agents with dishes performing greedily-optimally as outlined in docstring
        for player in players_holding_dishes:
            best_pickup_soup = None
            best_pickup_value = 0

            # find best soup to pick up with dish agent currently has
            for soup in non_idle_soup_vals:
                # How far away the soup is (inf if not-reachable)
                pickup_dist = mp.min_cost_to_feature(player.pos_and_or, [soup.position])

                # mask to award zero score if not reachable
                # Note: this means that potentially "useful" dish pickups (where agent passes dish to other agent
                # that can reach the soup) do not recive a potential bump
                is_useful = int(pickup_dist < np.inf)

                # Always assume worst-case discounting for step 4, and bump zero-valued soups to 1 as mentioned in docstring
                pickup_soup_value = gamma**potential_params['max_delivery_steps'] * max(self.get_recipe_value(state, soup.recipe), 1)
                cook_time_remaining = soup.cook_time - soup._cooking_tick
                discount = gamma**max(cook_time_remaining, min(pickup_dist, potential_params['max_pickup_steps']))

                # Final discount-adjusted value for this player pursuing this soup
                pickup_value = discount * pickup_soup_value * is_useful

                # Update best soup found for this player
                if pickup_dist < np.inf and pickup_value > best_pickup_value:
                    best_pickup_soup = soup
                    best_pickup_value = pickup_value
            
            # Set best-case score for this soup. Can only improve upon previous players policies
            # Note cooperative policies between players not considered
            if best_pickup_soup:
                non_idle_soup_vals[best_pickup_soup] = max(non_idle_soup_vals[best_pickup_soup], best_pickup_value)

        # Apply potential for each idle soup as calculated above
        for soup in non_idle_soup_vals:
            potential += non_idle_soup_vals[soup]



        ### Step 2 potential ###

        # Iterate over idle soups in decreasing order of value so we greedily prioritize higher valued soups
        for soup in idle_containers:
            # Calculate optimal recipe
            curr_recipe = Recipe(soup.ingredients)
            opt_recipe = self.get_optimal_possible_recipe(state, curr_recipe, discounted=True, potential_params=potential_params)

            # Calculate missing ingredients needed to complete optimal recipe
            missing_ingredients = list(opt_recipe.ingredients)
            for ingredient in soup.ingredients:
                missing_ingredients.remove(ingredient)

            # Base discount for steps 3-4
            discount = gamma**(max(potential_params['max_pickup_steps'], opt_recipe.time) + potential_params['max_delivery_steps'])

            # Add a multiplicative discount for each needed ingredient (this has the effect of giving more award to soups
            # that are closer to being completed)
            for ingredient in missing_ingredients:
                # Players who might have an ingredient we need
                pertinent_players = players_holding_ingredient[ingredient]
                dist = np.inf
                closest_player = None

                # Find closest player with ingredient we need
                for player in pertinent_players:
                    curr_dist = mp.min_cost_to_feature(player.pos_and_or, [soup.position])
                    if curr_dist < dist:
                        dist = curr_dist
                        closest_player = player

                # Update discount to account for adding this missing ingredient (defaults to min_coeff if no pertinent players exist)
                discount *= gamma**min(dist, potential_params['pot_{}_steps'.format(ingredient)])

                # Cross off this player's ingreident contribution so it can't be double-counted
                if closest_player:
                    pertinent_players.remove(closest_player)

            # Update discount to account for time it takes to start the soup cooking once last ingredient is added
            if missing_ingredients:
                # We assume it only takes one timestep if there are missing ingredients since the agent delivering the last ingredient
                # will be at the pot already
                discount *= gamma
            else:
                # Otherwise, we assume that every player holding nothing will make a beeline to this soup since it's already optimal
                cook_dist = min([mp.min_cost_to_feature(player.pos_and_or, [soup.position]) for player in players_holding_nothing], default=np.inf)
                discount *= gamma**min(cook_dist, potential_params['max_pickup_steps'])

            potential += discount * max(self.get_recipe_value(state, opt_recipe), 1)


        ### Step 1 Potential ###

        # Add potential for each ingredient that is left over after using all others to complete optimal recipes
        for elem in CFG_ALL_RAWFOOD:
            for player in players_holding_ingredient[elem]:
                dist = mp.min_cost_to_feature(player.pos_and_or, self.get_empty_containers(container_states))
                is_useful = int(dist < np.inf)
                discount = gamma**(min(potential_params[f'pot_{elem}_steps'], dist) + potential_params['max_pickup_steps'] + potential_params['max_delivery_steps']) * is_useful
                potential += discount * potential_params[f'{elem}_value']

        # At last
        return potential
