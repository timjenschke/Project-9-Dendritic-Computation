import numpy as np

class Mask:

    def __init__(self):
        pass

    @staticmethod
    def create_mask_dendrites_to_soma(num_dendrites,num_soma):
            """Creates mask for weigths between dendritic and soma layer
                Column represents one dendritic neuron
                Row represents the input of one soma neuron

            num_dendrites: number of dendritic neurons
            num_soma: number of soma neurons
            """
            if num_dendrites - num_soma < 0:
                raise ValueError("Number of dendrites must be greater than number of soma")

            mask = np.zeros((num_soma,num_dendrites),dtype=np.float32) 
            pos1 = list(np.arange(0,num_soma)) + list(np.random.choice(np.arange(0,num_soma),size=num_dendrites-num_soma))
            np.random.shuffle(pos1)
           
            for i in range(num_dendrites):
                row = pos1[i]
                mask[row][i] = 1
            return mask

    @staticmethod
    def create_mask_input_to_dendrites_exclusive(num_inputs, num_dendrites, typ):
            """Creates mask for weights between input and dendritic layer.
                Column represents one dendritic neuron
                Row represents all inputs in a flattened manner

                num_inputs: all inputs in a flattened array
                num_dendrites: number of dendritic neurons
                
                Each dendritic neuron receives num_inputs/num_dendrites inputs.
                !!! No input is feed into more than one dendrite !!!
            
            """
            if num_inputs%num_dendrites != 0:
                raise ValueError('Input neurons devided by dendrites must be an integer')

            mask = np.zeros((num_dendrites, num_inputs),dtype=np.float32)
            input_per_dendrite = int(num_inputs / num_dendrites)

            if typ == "random":
                all_pixels = np.arange(0,num_inputs)
                np.random.shuffle(all_pixels)
                counter=0
                for i in range(num_dendrites):
                    for j in range(counter, counter+input_per_dendrite):
                        pos = all_pixels[j]
                        mask[i][pos] = 1
                    counter += input_per_dendrite
            return mask
    
    @staticmethod
    def create_mask_input_to_dendrites(num_inputs, num_dendrites, typ, inputs_per_dendrite, image_length=28):
        """Creates mask for weights between input and dendritic layer.
                Column represents one dendritic neuron
                Row represents all inputs in a flattened manner

                num_inputs: all inputs in a flattened array
                num_dendrites: number of dendritic neurons
                
                An input pixel can be feed into several dendrites.
            
            """
        mask = np.zeros((num_dendrites, num_inputs),dtype=np.float32)
        pos = np.arange(0,num_inputs)

        if typ == "full":
            mask[mask == 0] = 1
        
        if typ == "random":
                for row in range(num_dendrites):
                    cols = np.random.choice(pos,size=inputs_per_dendrite, replace=False)
                    for col in cols:
                        mask[row][col] = 1

        if typ == "local":
            
            if num_inputs % image_length != 0:
                 raise ValueError('Image dimensions do not fit together.')

            image_depth = num_inputs//image_length   
            image = np.reshape(pos, (image_depth, image_length))

            col_pos = np.arange(2,image_length-2)
            row_pos = np.arange(2,image_depth-2)
            centroids = np.array([np.array([np.random.choice(row_pos), np.random.choice(col_pos)]) for _ in range(num_dendrites)])

            def get_neighbors(centroid):
                row = centroid[0]
                col = centroid[1]

                rows = np.arange(row-2,row+2+1)
                cols = np.arange(col-2,col+2+1)
                neighbors = np.dstack(np.meshgrid(rows, cols)).reshape(-1, 2) # cartesian product
                
                return neighbors

            # each dendrite gets one centroid and its receptive field assigned
            for dendrite, centroid in enumerate(centroids):
                neighbors = get_neighbors(centroid)
                
                # get the index of inputs_per_dendrite neighbors and collect the neighbor cordinates in selected_neighbors
                indexes = np.random.choice(np.arange(0,np.shape(neighbors)[0]), size=inputs_per_dendrite, replace=False)
                selected_neighbors = np.array([neighbors[index] for index in indexes])

                # image[row][col] is a unique number between 0 and 28*28 --> activate that neuron in mask for the respective dendrite
                for row, col in selected_neighbors:
                    column = image[row][col]
                    mask[dendrite][column] = 1


        return mask