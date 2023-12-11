from sigpro import contributing
import json
import inspect
from mlblocks.discovery import load_primitive
from mlblocks.mlblock import import_object

def make_primitive(primitive, primitive_type, primitive_subtype,  #No longer need to pass in primitive function assuming it is at the location in primitive name (to be imported)
                    primitive_function = None, #primitive_args,
                    context_arguments=None, fixed_hyperparameters=None,
                    tunable_hyperparameters=None, primitive_inputs = None, primitive_outputs=None):
    """Create a primitive JSON.

    During the JSON creation the primitive function signature is validated to
    ensure that it matches the primitive type and subtype implicitly specified
    by the primitive name.

    Any additional function arguments are also validated to ensure that the
    function does actually expect them.

    Args:
        primitive (str):
            The name of the primitive, the python path including the name of the
            module and the name of the function.
        primitive_type (str):
            Type of primitive.
        primitive_subtype (str):
            Subtype of the primitive.
        primitive_function (function):
            Function applied by the primitive.
        context_arguments (list or None):
            A list with dictionaries containing the name and type of the context arguments.
        fixed_hyperparameters (dict or None):
            A dictionary containing as key the name of the hyperparameter and as
            value a dictionary containing the type and the default value that it
            should take.
        tunable_hyperparameters (dict or None):
            A dictionary containing as key the name of the hyperparameter and as
            value a dictionary containing the type and the default value and the
            range of values that it can take.
        primitive_inputs (list or None):
            A list with dictionaries containing the name and type of the input values. If
            ``None`` default values for those will be used.
        primitive_outputs (list or None):
            A list with dictionaries containing the name and type of the output values. If
            ``None`` default values for those will be used.

    Raises:
        ValueError:
            If the primitive specification arguments are not valid.

    Returns:
        dict:
            Generated JSON file as a python dictionary
    """
    context_arguments = context_arguments or []
    fixed_hyperparameters = fixed_hyperparameters or {}
    tunable_hyperparameters = tunable_hyperparameters or {}

    primitive_spec = contributing._get_primitive_spec(primitive_type, primitive_subtype)
    primitive_inputs = primitive_inputs  or primitive_spec['args']
    primitive_outputs = primitive_outputs or primitive_spec['output']

    if primitive_function == None:
        primitive_function = import_object(primitive)

    primitive_args = contributing._get_primitive_args(
        primitive_function,
        primitive_inputs,
        context_arguments,
        fixed_hyperparameters,
        tunable_hyperparameters
    )

    primitive_dict = {
        'name': primitive,
        'primitive': primitive,
        'classifiers': {
            'type': primitive_type,
            'subtype': primitive_subtype
        },
        'produce': {
            'args': primitive_args,
            'output': [
                {
                    'name': primitive_output['name'],
                    'type': primitive_output['type'],
                }
                for primitive_output in primitive_outputs
            ],
        },
        'hyperparameters': {
            'fixed': fixed_hyperparameters,
            'tunable': tunable_hyperparameters
        }
    }
    return primitive_dict


class Primitive: 

    def __init__(self, primitive, primitive_type, primitive_subtype, 
                primitive_function = None, init_params = {}):

        """
        Initialize primitive object. 
        """
        self.primitive = primitive
        self.primitive_type = primitive_type
        self.primitive_subtype = primitive_subtype
        self.tunable_hyperparameters = {}
        self.fixed_hyperparameters = {}
        self.context_arguments = []
        primitive_spec = contributing._get_primitive_spec(primitive_type, primitive_subtype)
        self.primitive_inputs = primitive_spec['args']
        self.primitive_outputs = primitive_spec['output']

        contributing._check_primitive_type_and_subtype(primitive_type, primitive_subtype)

        if primitive_function == None:
            primitive_function = import_object(primitive)
        self.primitive_function = primitive_function
        self.hyperparameter_values = init_params #record the init_param values as well.

    def get_name(self):
        return self.primitive
    def get_inputs(self):
        return self.primitive_inputs.copy()
    def get_outputs(self):
        return self.primitive_outputs.copy()

    def make_primitive_json(self): #return primitive json.
        self._validate_primitive_spec()
        return make_primitive(self.primitive, self.primitive_type, self.primitive_subtype, self.primitive_function , self.context_arguments, self.fixed_hyperparameters, self.tunable_hyperparameters, self.primitive_outputs)

    def write_primitive_json(self, primitives_path = 'sigpro/primitives', primitives_subfolders=True):

        """
        primitives_path (str):
            Path to the root of the primitives folder, in which the primitives JSON will be stored.
            Defaults to `sigpro/primitives`.
        primitives_subfolders (bool):
            Whether to store the primitive JSON in a subfolder tree (``True``) or to use a flat
            primitive name (``False``). Defaults to ``True``.
        """
        pj = self.make_primitive_json()
        contributing._write_primitive(pj, self.primitive, primitives_path, primitives_subfolders)

    def _validate_primitive_spec(self): #check if the primitive is actually up-to-spec for debugging use/use in pipelines; throws appropriate errors.
        
        self.primitive_args = _get_primitive_args(
            self.primitive_function,
            self.primitive_inputs,
            self.context_arguments,
            self.fixed_hyperparameters,
            self.tunable_hyperparameters
        )
        pass
    
    
    def get_hyperparam_dict(self, name = None):
        """
        Return the dictionary of parameters (for use in larger pipelines such as Linear, etc)
        """
        if name == None:
            name = self.primitive
        return { 'name': name, 'primitive': self.primitive, 'init_params': self.hyperparameter_values}



    def set_primitive_function(self, primitive_function):
        self.primitive_function = primitive_function

    def set_primitive_inputs(self, primitive_inputs): #does the user really need to specify this?
        self.primitive_inputs = primitive_inputs
            
    def set_primitive_outputs(self, primitive_outputs): #does the user really need to specify this?
        self.primitive_outputs = primitive_outputs

    def _set_primitive_type(self, primitive_type):
        self.primitive_type = primitive_type
    def _set_primitive_subtype(self, primitive_subtype):
        self.primitive_subtype = primitive_subtype

    def set_context_arguments(self, context_arguments):
        self.context_arguments = context_arguments

    def set_tunable_hyperparameters(self, tunable_hyperparameters):
        self.tunable_hyperparameters = tunable_hyperparameters

    def set_fixed_hyperparameters(self, fixed_hyperparameters):
        self.fixed_hyperparameters = fixed_hyperparameters

    def add_context_arguments(self, context_arguments):
        for arg in context_argments:
            if arg not in self.context_arguments:
                context_arguments.append(arg)
    def add_fixed_hyperparameter(self, hyperparams):
        for hyperparam in hyperparams:
            self.fixed_hyperparameters[hyperparam] = hyperparams[hyperparam]
    def add_tunable_hyperparameter(self, hyperparams):
        for hyperparam in hyperparams:
            self.tunable_hyperparameters[hyperparam] = hyperparams[hyperparam]
    def remove_context_arguments(self, context_arguments):
        for arg in context_argments:
            if arg in self.context_arguments:
                context_arguments.remove(arg)
    def remove_fixed_hyperparameter(self, hyperparams):
        for hyperparam in hyperparams:
            del self.fixed_hyperparameters[hyperparam]
    def remove_tunable_hyperparameter(self, hyperparams):
        for hyperparam in hyperparams:
            del self.tunable_hyperparameters[hyperparam]



class TransformationPrimitive(Primitive):

    def __init__(self, primitive, primitive_subtype,  init_params = {}):
        super().__init__(primitive, 'transformation',primitive_subtype, init_params = init_params)

    pass

class AmplitudeTransformation(TransformationPrimitive):

    def __init__(self, primitive, init_params = {}):
        super().__init__(primitive, 'amplitude', init_params = init_params)

    pass


class FrequencyTransformation(TransformationPrimitive):

    def __init__(self, primitive, init_params = {}):
        super().__init__(primitive,  'frequency', init_params = init_params)

    pass

class FrequencyTimeTransformation(TransformationPrimitive):

    def __init__(self, primitive, init_params = {}):
        super().__init__(primitive, 'frequency_time', init_params = init_params)




class ComparativeTransformation(TransformationPrimitive):
    pass


class AggregationPrimitive(Primitive):
    def __init__(self, primitive, primitive_subtype, init_params = {}):
        super().__init__(primitive, 'aggregation', primitive_subtype, init_params = init_params)


class AmplitudeAggregation(AggregationPrimitive):

    def __init__(self, primitive,  init_params = {}):
        super().__init__(primitive, 'amplitude', init_params = init_params)

class FrequencyAggregation(AggregationPrimitive):

    def __init__(self, primitive,  init_params = {}):
        super().__init__(primitive,  'frequency',  init_params = init_params)

class FrequencyTimeAggregation(AggregationPrimitive):

    def __init__(self, primitive, init_params = {}):
        super().__init__(primitive, 'frequency_time', init_params = init_params)


class ComparativeAggregation(AggregationPrimitive):
    pass

def primitive_from_json(json_path,init_params = {}): #Create the proper primitive object given an input json, primitive_functio and hyperparameters.
    primitive_dict = load_primitive(json_path)

    primitive = primitive_dict['primitive']
    primitive_type = primitive_dict['classifiers']['type']
    primitive_subtype = primitive_dict['classifiers']['subtype']

    return Primitive(primitive, primitive_type, primitive_subtype, init_params = init_params)
    