run = 'pipe_test' # for filenames during saving of results

network_type = 'wikipedia' # wikipedia, synthetic_ER, synthetic_BA
size = 'small' # size of dataset

feature = betti_numbers # nx.average_clustering, betti_numbers, compressibility
steps_per_episode = 10 # average KNOT session has ~9 unique node visits

base_path = '/content/drive/My Drive/GraphRL_v2/'

data_load_path = os.path.join(base_path, 'Environments', network_type + '_' + size + '.json')
with open(data_load_path, 'r') as f:
  all_data = json.load(f)

train_data = all_data['train']
val_data = all_data['val']
test_data = all_data['test']

save_folder = base_path + network_type + '_' + size + '_' + feature.__name__ + '_run_' + run

if not os.path.isdir(save_folder):
  os.makedirs(save_folder)
  print('Created folder:', save_folder)

else:
  print('Folder exists. ')

# build training environments

train_graphs = []
train_environments = []

for idx in range(len(train_data)):

  base_G = nx.node_link_graph(train_data[str(idx)])
  base_G = node_defeaturizer(base_G)
  train_graphs.append(base_G)

  G = node_featurizer(base_G, mode = 'LDP')
  environment = GraphEnvironment(idx, G, steps_per_episode, feature)
  train_environments.append(environment)

train_environments = MultipleEnvironments(train_environments)

# build validation environments

val_graphs = []
val_environments = []

for idx in range(len(val_data)):

  base_G = nx.node_link_graph(val_data[str(idx)])
  base_G = node_defeaturizer(base_G)
  val_graphs.append(base_G)

  G = node_featurizer(base_G, mode = 'LDP')
  environment = GraphEnvironment(idx, G, steps_per_episode, feature)
  val_environments.append(environment)

val_environments = MultipleEnvironments(val_environments)

# build test environments

test_graphs = []
test_environments = []

for idx in range(len(test_data)):

  base_G = nx.node_link_graph(test_data[str(idx)])
  base_G = node_defeaturizer(base_G)
  test_graphs.append(base_G)

  G = node_featurizer(base_G, mode = 'LDP')
  environment = GraphEnvironment(idx, G, steps_per_episode, feature)
  test_environments.append(environment)

test_environments = MultipleEnvironments(test_environments)