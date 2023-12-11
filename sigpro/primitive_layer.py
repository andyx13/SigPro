from sigpro import contributing, primitive, linear_pipeline
import json
import inspect
from mlblocks.discovery import load_primitive, MLPipeline
from mlblocks.mlblock import import_object
from collections import Counter


class TreeTransformationLayer: #TBD -- simple rep in terms of the primitives in the layer.
    pass



### Todo: 1. make it a bit easier for the user to init stuff
### Todo: 2. (possible) try to rewrite the internal nomenclature??

class PrimitiveLayer:


        """
        transformations is a list of transformations

        input_columns is a list of dictionaries mapping the basic primitive inputs as keys to the data-format primitive input strings as values
        output_columns is similar


        example input:

        [fft_prim1, fftreal_prim2],     [ [{amplitude_values: av1 , frequency: fr1}, {amplitude_values: av2, frequency: fr2}], [{amplitude_values: av1 , frequency: fr1}]   ]

        'prim_basename' + 'layer # and index #' + '.output_name' without letting user pass those in; then the task is to just feed in the proper input names. 
        """

    def __init__(self, primitives, input_columns): 

        if len(primitives) != len(input_columns):
            raise ValueError('transformations and input_columns should have the same length')

        self.layer_index = 1 #will be essential for uniquely naming each primitive

        counter = Counter()

        self.primitives = primitives.copy()
        self.repetition_count = []

        for primitive in primitives:
            Counter[primitive.get_name()] += 1
            self.repetition_count.append(Counter[primitive.get_name()])

        self.input_columns = input_columns.copy()


    pass

    def _get_expanded_inputs(self, rep_counts = False):

        expanded_primitives = []
        expanded_inputs = []
        expanded_rep_counts = []

        for i, primitive in enumerate(self.primitives):
            for j, mapping in enumerate(self.input_columns):
                expanded_primitives.append(primitive)
                expanded_inputs.append(mapping.copy())
                expanded_rep_counts.append(self.repetition_count[i])

        if rep_counts:
            return expanded_primitives, expanded_inputs, expanded_rep_counts
        else:
            return expanded_primitives, expanded_inputs

    def get_input_column_dict(self): #return the input columns

        input_cols = {}
        expanded_primitives, expanded_inputs, rep_counts = self._get_expanded_inputs(rep_counts = True)
        for i, primitive in enumerate(expanded_primitives):
            input_cols[self._get_primitive_str(primitive, rep_counts[i])] = expanded_inputs[i]


         return input_cols


    def get_output_column_dict(self):

        output_cols = {}
        expanded_primitives, expanded_inputs, rep_counts = self._get_expanded_inputs(rep_counts = True)
        for i, primitive in enumerate(expanded_primitives):

            primitive_name = primitive.get_name()
            primitive_outputs  = primitive.get_outputs()
            output_cols[self._get_primitive_str(primitive, rep_counts[i])] = {}

            for output_dict in primitive_outputs:
                output_str = self._get_primitive_str(primitive, rep_counts[i]) + '.' + output_dict['name']
                output_cols[self._get_primitive_str(primitive, rep_counts[i])][output_dict['name']] = output_str  

        return output_cols
    def get_init_params_dict(self):

        init_params = {}
        expanded_primitives, _ , rep_counts= self._get_expanded_inputs(rep_counts = True)
        for i, primitive in enumerate(expanded_primitives):

            init_params[self._get_primitive_str(primitive, rep_counts[i])] = primitive.get_hyperparam_dict(self._get_primitive_str(primitive, rep_counts[i]))['init_params'].copy()

        return init_params

    def _get_primitive_str(self, primitive, primitive_index):
        primitive_str = primitive.get_name()
        return primitive_str + f'_Layer#{self.layer_index}_#{primitive_index}'

    def set_layer_index(self, layer_index):
        self.layer_index = layer_index

    def set_primitive_inputs(self, input_columns):

        """

        """
        if not isinstance(input_columns, list):
            raise ValueError('input_columns should be a list')
        elif len(input_columns) != len(self.primitives):
            raise ValueError('primitives and input_columns should have the same length')
        else:
            self.input_columns = input_columns.copy()
    
    def _generate_primitives_list(self, rep_count = True): #do not need the repetition count here 
        
        if rep_count:
            expanded_primitives, expanded_inputs, rep_counts  = self._get_expanded_inputs(rep_counts = True)
            return [(primitive.get_name(), rep_counts[i]) for i, primitive in enumerate(expanded_primitives)]
        else:
            expanded_primitives, expanded_inputs = self._get_expanded_inputs(rep_counts = False)
            return [primitive.get_name() for primitive in expanded_primitives]

    def _generate_primitive_names_list(self):
        expanded_primitives, expanded_inputs, rep_counts  = self._get_expanded_inputs(rep_counts = True)
        return [self._get_primitive_str(primitive, rep_counts[i]) for i, primitive in enumerate(expanded_primitives)]


    def _generate_initial_parameters_list(self):
        expanded_primitives, expanded_inputs, rep_counts  = self._get_expanded_inputs(rep_counts = True)
        return [primitive.get_hyperparam_dict(self._get_primitive_str(primitive, rep_counts[i])) for i, primitive in enumerate(expanded_transformations)]

    def convert_prefix_dictionary(self, input_prefix_dict, add_output_name = False):

        output_prefix_dict = {}

        expanded_primitives, expanded_inputs, rep_counts  = self._get_expanded_inputs(rep_counts = True)


        for i, primitive in enumerate(expanded_primitives):

            primitive_name = primitive.get_name()
            primitive_outputs  = primitive.get_outputs()

            if input_prefix_dict is not None:
                if 'amplitude_values' in expanded_inputs[i]:
                    expanded_input_prefix = input_prefix_dict[expanded_inputs[i]['amplitude_values']]
                else:
                    expanded_input_prefix = input_prefix_dict[sorted(list(expanded_inputs[i].values()))[0]]
            else:
                expanded_input_prefix = ''

            for output_dict in primitive_outputs:
                output_str = self._get_primitive_str(primitive, rep_counts[i]) + '.' + output_dict['name']
                optional_output = ''
                if add_output_name:
                    optional_output = '.' + output_dict['name']


                output_prefix_dict[output_str] = expanded_input_prefix + '.' + primitive_name.split('.')[-1] + str(rep_counts[i]) + optional_output
                #output_cols[self._get_primitive_str(primitive, rep_counts[i])][output_dict['name']] = output_str  

        return output_prefix_dict


#TODO: Can probably combine transformationlayer and aggregationlayer into a single primitive_layer class and subclass off of that.

class TransformationLayer(PrimitiveLayer): 


    """

    This class is largely technical in order to enable the creation of layer pipelines. 


    #NOTE: CAN PASS IN INPUT NAMES WHEN CREATING THE MLPIPELINE!

        def _get_block_args(self, block_name, block_args, context):
            Get the arguments expected by the block method from the context.

            The arguments will be taken from the context using both the method
            arguments specification and the ``input_names`` given when the pipeline
            was created.

            ...


    input_names (dict):  dictionary that maps input variable names with the actual names expected by each primitive. This allows reusing the same input argument for multiple primitives that name it differently, as well as passing different values to primitives that expect arguments named similary.

    output_names (dict): dictionary that maps output variable names with the name these variables will be given when stored in the context dictionary. This allows storing the output of different primitives in different variables, even if the primitive output name is the same one.

    """
    def __init__(self, transformations, input_columns): 
        super().__init__(transformations, input_columns)
        self.layer_type = 'transformation'

class AggregationLayer(PrimitiveLayer):

    def __init__(self, aggregations, input_columns):
        super().__init__(aggregations, input_columns)
        self.layer_type = 'aggregation'

class LayerPipeline(linear_pipeline.Pipeline):

    '''
    Idea (for tree): the user passes in a list of layers, which are simple graph. We can even allow creation w treepipeline using a listoflists and a list.
    Idea (for general): the user passes a list of layers, which are more complicated.

    From this, the module itself will build the layers one by one (inferring the proper input arguments from above, flagging any incorrect connections.

    Todo: how do we ensure the outputs have the right nomenclature?



    the goal is to try to make it easier for the user to input an arbitrary dag.


    '''
    def __init__(self, transformation_layers, aggregation_layer,):

        self.transformation_layers = transformation_layers

        for i in range(len(self.transformation_layers)):
            self.transformation_layers[i].set_layer_index(i+1)
        self.aggregation_layer = aggregation_layer
        self.aggregation_layer.set_layer_index(len(self.transformation_layers)) + 1)
        self.expanded_primitive_list = self._generate_primitives_list()
        self.expanded_primitive_name_list = self._generate_primitive_names_list()

    def _generate_output_feature_names(self): #generate the list of output feature names in t1.t2.t6.a2 format

        pass

    def get_primitive_object(self, primitive_name):

        pass 

    def _generate_primitives_list(self):
        primitive_list = []
        for transformation_layer in self.transformation_layers:
            primitive_list+= transformation_layer._generate_primitives_list()

        primitive_list += self.aggregation_layer._generate_primitives_list()
        return primitive_list

    def _generate_primitive_names_list(self):
        primitive_names_list = []
        for transformation_layer in self.transformation_layers:
            primitive_names_list+= transformation_layer._generate_primitive_names_list()

        primitive_names_list += self.aggregation_layer._generate_primitive_names_list()
        return primitive_names_list


    """
    output_names: ALL primitives.
    outputs: ONLy attached to aggregations.
    """


    def get_full_output_names(self, column_remapping):

        prefix_dict = None
        for transformation_layer in self.transformation_layers:
            prefix_dict = transformation_layer.convert_prefix_dictionary(prefix_dict)

        prefix_dict = self.aggregation_layer.convert_prefix_dictionary(prefix_dict, add_output_name = True)


        #Next, we have to tweak the output_names
        remapped_prefix_dict = {}
        for key in prefix_dict:
            agg_name = key.split('.')[-1]
            base_name = key[-1 * (len(agg_name) + 1)]

            remapped_prefix_dict[column_remapping[base_name] + '.' + agg_name] = prefix_dict[key]

        return {'default': [{'name': remapped_prefix_dict[key], 'variable': key} for key in remapped_prefix_dict]}
        
        

    def _build_pipeline(self): #do some dict manipulation here. 


        #for transformation_layer in 

        final_primitives = self.expanded_primitive_list.copy()
        init_params = {}
        input_names = {}
        output_names = {}  #we get these from the layers themselves
        outputs_final = self.get_full_output_names()
        primitive_names = self.expanded_primitive_name_list.copy()


        for transformation_layer in transformation_layers:
            input_column_dict = transformation_layer.get_input_column_dict()
            output_column_dict  = transformation_layer.get_output_column_dict()
            init_param_dict = transformation_layer.get_init_params_dict()
            for key in input_column_dict:
                input_names[key] = input_column_dict[key]
            for key in output_column_dict:
                output_names[key] = output_column_dict[key]
            for key in init_param_dict:
                init_params[key] = init_param_dict[key]
            
            



        #agg  
        #update the input_names dict of adds
        #update the output names dict of affs
        #update the init_params dict of aggs

        input_column_dict = self.aggregation_layer.get_input_column_dict()
        output_column_dict = self.aggregation_layer.get_output_column_dict()
        init_param_dict = self.aggregation_layer.get_init_params_dict()
        for key in input_column_dict:
            input_names[key] = input_column_dict[key]
        for key in output_column_dict:
            output_names[key] = output_column_dict[key]
        for key in init_param_dict:
            init_params[key] = init_param_dict[key]

        
    #This isn't fully correct -- unfortunately only one of pipeline, primitives can be a list of strings! 
    #We will have to manually specify the mapping from OUR primitive names to the sigpro-counter primitive names by listifying them both and making a dict.
    #Then we have to update all the appropriate keys in init_params, input_names, output_names.



        primitive_counter = Counter()
        column_remapping = {}
        for i in range(len(final_primitives)):
            primitive_counter[final_primitives[i]] += 1
            column_remapping[primitive_names[i]] = f'{final_primitives[i]}#{primitive_counter}'

        

        init_params_final = {}
        input_names_final = {}
        output_names_final = {}
        for key in init_params:
            init_params_final[column_remapping[key]] = init_params[key]
        for key in input_names:
            input_names_final[column_remapping[key]] = input_names[key]
        for key in output_names:
            output_names_final[column_remapping[key]] = output_names[key]
        

        return MLPipeline( 
            primitives = final_primitives,
            init_params = init_params_final,
            input_names = input_names_final,
            output_names = output_names_final,
            outputs = outputs_final
        )


    pass


