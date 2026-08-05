from utils import *

# CLASSES

class Device:
    def __init__(self, id, ram, storage, cpu, bandwidth, battery, charging):
        self.id = id
        self.ram = ram
        self.storage = storage
        self.cpu = cpu
        self.bandwidth = bandwidth
        self.battery = battery
        self.charging = charging
        self.model: keras.Sequential = Server.create_model()
        self.last_round_participated = 0
        self.data = None  # Placeholder for dataset partition
        self.test_data = None
        self.number_of_times_fitted = 0
        self.hardware_value_sum = 0.0
        
    def lose_battery(self):
        if self.hardware_value_sum > 0.3:
            self.hardware_value_sum -= 0.3
        else:
            self.hardware_value_sum = 0
            print("device turned off!")
        
        # if float(self.battery) > 0.3:
        #     self.battery -= 0.3
        # else:
        #     self.battery = 0
        #     print("device turned off!")

class Server:
    def __init__(self, devices_list: list[Device]):
        self.model: keras.Sequential = Server.create_model()
        self.current_learning_iteration = 0
        self.LAST_WEIGHTS_SENT_FOR_ALL_DEVICES = []
        self.x_test_global = []
        self.y_test_global = []
        self.devices = devices_list
               
        # The variables below are used to keep track of the remaining rounds and generations and to store the variances
        # They do not logically belong to the Server class
        self.remaining_generation = 0
        self.remaining_round = 0
        self.variances = []  # List to store variances as [round, gen, solution, variance]
        self.pareto_fronts = [] # List to store pareto fronts as [round, gen, solution]

    def evaluate(self, x_test=None, y_test=None, verbose = 0):
        if x_test is None and y_test is None:
            test_loss, test_acc = self.model.evaluate(self.x_test_global, self.y_test_global, verbose)
            return test_loss, test_acc
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=verbose)
        return test_loss, test_acc

    def get_weights(self):
        return self.model.get_weights()

    def set_aggregated_weight(self):
        self.model.set_weight(Server.aggregate_weights())

    def give_global_model_weights_to_bitstring_devices(self, bitstring):
        for device in self.devices:
            if int(bitstring[int(device.id)]) == 1:
                device.model.set_weights(self.model.get_weights())

    def create_model() -> keras.Sequential:
        model = keras.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                        # new
                        loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def aggregate_weights(self, bitstring):
        """Computes the weighted average of model weights from all devices and updates the global model."""
        def sum_all_nested_lists(list_of_lists):
            def recursive_sum(lists):
                if isinstance(lists[0], list):
                    return [recursive_sum([lst[i] for lst in lists]) for i in range(len(lists[0]))]
                else:
                    return sum(lists)

            return recursive_sum(list_of_lists)

        def multiply_nested_list(lst, factor):
            result = []
            for item in lst:
                if isinstance(item, list):
                    # Recursively handle sublists
                    result.append(multiply_nested_list(item, factor))
                else:
                    # Multiply number
                    result.append(item * factor)
            return result

        selected_devices: list[Device] = []
        for device in self.devices:
            if int(bitstring[int(device.id)]) == 1:
                selected_devices.append(device)

        num_devices = len(selected_devices)
        if num_devices == 0:
            print("No devices available for aggregation.")
            return

        # device_participation_ratio = []
        
        # device_participation_frequency_reduction = [] # to support lesser participated clients
        
        data_lengths = []

        for device in selected_devices:
            # print("*******************")
            # print(device.id)
            # device_participation_ratio.append(device.last_round_participated / self.current_learning_iteration)
            # print("this device's participation ratio:")
            # print(device.last_round_participated / self.current_learning_iteration)
            data_lengths.append(len(device.data[0]))
            # print("this device's data to all ratio:")
            # print(len(device.data[0])/60000.0)
            
            # device_participation_frequency_reduction.append(1.0/device.number_of_times_fitted)


        sum_data = 0
        for data_len in data_lengths:
            sum_data += data_len

        data_fractions = []
        for device in selected_devices:
            data_fractions.append(len(device.data[0])/float(sum_data))



        # combined_weights = [fraction * ratio * participation_frequency_reduction_ratio for fraction, ratio, participation_frequency_reduction_ratio in zip(data_fractions, device_participation_ratio, device_participation_frequency_reduction)]
        total_weight = sum(data_fractions)
        # normalized_weights = [w / total_weight for w in combined_weights]
        normalized_weights = data_fractions
        # print(normalized_weights)


        aggregated_weights_devices = []
        for d in range(len(selected_devices)):
            # aggregated_weights_devices.append(multiply_nested_list(selected_devices[d].model.get_weights(), data_fractions[d]*device_participation_ratio[d]))
            aggregated_weights_devices.append(multiply_nested_list(self.LAST_WEIGHTS_SENT_FOR_ALL_DEVICES[int(selected_devices[d].id)], normalized_weights[d]))

        aggregated_weights = sum_all_nested_lists(aggregated_weights_devices)

        return aggregated_weights


class NodeSelectionProblem(Problem):
    def __init__(self, devices: list[Device], server: Server, max_number_of_generations, max_number_of_rounds):
        super().__init__(
            n_var=len(devices),         # Number of variables (bitstring length)
            n_obj=3,                   # Number of objectives
            n_constr=0,                # No constraints
            xl=np.zeros(len(devices)),  # Lower bound (0)
            xu=np.ones(len(devices)),   # Upper bound (1)
            type_var=np.bool_          # Binary variables (bitstrings)
        )
        self.MAX_NUMBER_OF_GENERATIONS = max_number_of_generations
        self.MAX_NUMBER_OF_ROUNDS = max_number_of_rounds
        self.devices = devices
        self.server = server
        self.x_test_global = server.x_test_global
        self.y_test_global = server.y_test_global

        # Save the initial global model weights
        self.initial_global_weights = server.get_weights()

    def _evaluate(self, X, out, *args, **kwargs):
        MAX_NUMBER_OF_GENERATIONS = self.MAX_NUMBER_OF_GENERATIONS
        MAX_NUMBER_OF_ROUNDS = self.MAX_NUMBER_OF_ROUNDS
        """Evaluates objective values for each solution in the population."""
        num_solutions = len(X)
        F = np.zeros((num_solutions, 3))  # Initialize objective matrix

        for i, bitstring in enumerate(X):
            # print(f"evaluating: {bitstring}")
            # TODO: check bitstring type
            # Reset the global model to its initial state
            # Update device participation based on the bitstring
            selected_devices = [device for device, bit in zip(self.devices, bitstring) if int(bit) == 1]

            # Objective 1: Hardware Objectives (maximize)
            hardware_overhead = 0.0
            for device in selected_devices:
                device_hardware_overhead = float(6 - (device.hardware_value_sum)) / 6.0
                hardware_overhead += device_hardware_overhead

            F[i, 0] = round(hardware_overhead/len(selected_devices), 2)  # Minimize (positive of hardware overhead)  # Added (/Selected Devices) to normalize between 0 and 1

            # Objective 2: Fairness (maximize)
            fairness_score = 0
            list_of_global_model_on_device_test_data_accuracies = []
            for device in self.devices:
                if int(bitstring[int(device.id)]) == 1:
                    _, accuracy = self.server.evaluate(device.test_data[0], device.test_data[1], verbose=0)
                    accuracy = round(accuracy, 2)
                    list_of_global_model_on_device_test_data_accuracies.append(accuracy)
                    fairness_score += accuracy
                    
            # round_to_save = MAX_NUMBER_OF_ROUNDS - self.server.remaining_round
            # generation_to_save = MAX_NUMBER_OF_GENERATIONS - self.server.remaining_generation
            # solution_to_save = str(bitstring).replace('True', '1').replace('False', '0').replace('  ',' ').replace('[ ','[').replace('\n', '').replace(' ', ',')
            # variance_to_save = round(1.0 / float(np.var(list_of_global_model_on_device_test_data_accuracies)), 2)
            # self.server.variances.append([round_to_save, generation_to_save, solution_to_save, variance_to_save])
            
            # with open("variances.txt", 'a') as f:
            #     f.write(f"{str(bitstring).replace('True', '1').replace('False', '0').replace('  ',' ').replace('[ ','[').replace('\n', '').replace(' ', ',')}\nround: {MAX_NUMBER_OF_ROUNDS - self.server.remaining_round} gen: {MAX_NUMBER_OF_GENERATIONS - self.server.remaining_generation} variance: {round(1.0 / float(np.var(list_of_global_model_on_device_test_data_accuracies)), 2)}\n")

            

            F[i, 1] = round(1.0 - (fairness_score/float(len(selected_devices))), 2)  # Minimize (negative of fairness score)  # Added (/Selected Devices) to normalize between 0 and 1

            # Objective 3: Global Model Accuracy (Performance) (maximize)
            temp_global_model = Server.create_model()
            temp_global_model.set_weights(self.performance_objective_aggregation(selected_devices))
            _, global_accuracy = temp_global_model.evaluate(self.server.x_test_global, self.server.y_test_global, verbose=0)
            F[i, 2] = round(1 - global_accuracy, 2)  # Minimize (1 - accuracy)

        self.server.remaining_generation -= 1
        out["F"] = F  # Set the objective values

    def performance_objective_aggregation(self, selected_devices: list[Device]):

        def sum_all_nested_lists(list_of_lists):
            def recursive_sum(lists):
                if isinstance(lists[0], list):
                    return [recursive_sum([lst[i] for lst in lists]) for i in range(len(lists[0]))]
                else:
                    return sum(lists)

            return recursive_sum(list_of_lists)

        def multiply_nested_list(lst, factor):
            result = []
            for item in lst:
                if isinstance(item, list):
                    # Recursively handle sublists
                    result.append(multiply_nested_list(item, factor))
                else:
                    # Multiply number
                    result.append(item * factor)
            return result

        num_devices = len(selected_devices)
        if num_devices == 0:
            print("No devices available for aggregation.")
            return

        device_weights_all_layers = []
        device_participation_ratio = []
        
        # device_participation_frequency_reduction = [] # to support lesser participated clients
        
        # device_participation_weights = []
        data_lengths = []

        for device in selected_devices:
            device_weights_all_layers.append(self.server.LAST_WEIGHTS_SENT_FOR_ALL_DEVICES[int(device.id)])
            # print("*******************")
            # print(device.id)
            device_participation_ratio.append(device.last_round_participated / self.server.current_learning_iteration)
            print(f"Device: {device.id} | Participation Ratio: {device.last_round_participated / self.server.current_learning_iteration}")
            # device_participation_weights.append(device.last_round_participated)
            # print("this device's participation ratio:")
            # print(device.last_round_participated / self.server.current_learning_iteration)

            data_lengths.append(len(device.data[0]))
            # print("this device's data to all ratio:")
            # print(len(device.data[0])/60000.0)
            
            
            # device_participation_frequency_reduction.append(1.0/device.number_of_times_fitted)

        sum_data = 0
        for data_len in data_lengths:
            sum_data += data_len

        # data_weights = []
        data_fractions = []
        for device in selected_devices:
            data_fractions.append(len(device.data[0])/float(sum_data))
            # data_weights.append(len(device.data[0]))


        # combined_weights = [fraction * ratio * participation_frequency_reduction_ratio for fraction, ratio, participation_frequency_reduction_ratio in zip(data_fractions, device_participation_ratio, device_participation_frequency_reduction)]
        combined_weights = [fraction * ratio for fraction, ratio in zip(data_fractions, device_participation_ratio)]
        total_weight = sum(combined_weights)
        normalized_weights = [w / total_weight for w in combined_weights]
        # print(normalized_weights)


        aggregated_weights_devices = []
        for d in range(len(selected_devices)):
            aggregated_weights_devices.append(multiply_nested_list(self.server.LAST_WEIGHTS_SENT_FOR_ALL_DEVICES[int(selected_devices[d].id)], normalized_weights[d]))


        aggregated_weights = sum_all_nested_lists(aggregated_weights_devices)

        # print("Aggregated weights:")
        # for layer_idx, layer_weights in enumerate(aggregated_weights):
        #     print(f"Layer {layer_idx}: {layer_weights.shape}")
        return aggregated_weights


# Functions

def fit_bitstring_devices(bitstring, server: Server, epochs=7):
    '''
    server: for using its "current_learning_iteration" variable
    '''

    server.current_learning_iteration += 1
    for device in server.devices:
        if int(bitstring[int(device.id)]) == 1:
            # TODO:
            # makes it so that the selection might choose a device that's been turned off
            # if the device is off, don't fit, use old weights saved on the server.
            # if the device is on, fit, update the weights saved on the server.
            # TODO:
            # COMMENTED FOR NOW, SINCE IT ALREADY AFFECTS NSGA2 EVALUATION
            
            # if device.hardware_value_sum :
            #     continue
            
            device.lose_battery()
            
            device.model.fit(device.data[0], device.data[1], epochs=epochs, verbose=0)
            # print(device.id)
            device.last_round_participated = server.current_learning_iteration
            server.LAST_WEIGHTS_SENT_FOR_ALL_DEVICES[int(device.id)] = device.model.get_weights()
            device.number_of_times_fitted += 1




def niid_labeldir_split(x_data, y_data, num_clients, beta, seed=None):
    num_classes = 10
    y_indices = np.array([np.argmax(label) for label in y_data])  # From one-hot to class index
    
    rng = np.random.default_rng(seed)  # Local random generator

    # Prepare client partitions
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(y_indices == k)[0]
        rng.shuffle(idx_k)

        # Dirichlet distribution for class k
        proportions = rng.dirichlet(np.repeat(beta, num_clients))

        # Scale proportions to match the number of available samples
        proportions = np.array([int(p * len(idx_k)) for p in proportions])
        # Fix total due to rounding
        while sum(proportions) < len(idx_k):
            proportions[np.argmin(proportions)] += 1
        while sum(proportions) > len(idx_k):
            proportions[np.argmax(proportions)] -= 1

        start = 0
        for i in range(num_clients):
            size = proportions[i]
            client_indices[i].extend(idx_k[start:start + size])
            start += size

    return client_indices



def random_hardware_value_for_devices(devices: list[Device]):
    random_values = [3.91, 0.62, 1.79, 4.96, 5.87, 2.14, 1.41, 5.18, 2.80, 3.00, 0.20, 1.02, 5.73, 0.69, 4.27,
                     5.37, 1.62, 0.93, 3.61, 2.90, 4.53, 2.13, 3.01, 0.07, 1.34, 3.90, 0.28, 1.89, 5.95, 2.76]
    
    for idx in range(len(devices)):
        devices[idx].hardware_value_sum = random_values[idx]
    
    print("Successfully gave each device a random value between 0 and 6 for its hardware objective!")
