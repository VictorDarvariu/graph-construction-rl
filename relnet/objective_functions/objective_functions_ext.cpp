#include <iostream>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graph_utility.hpp>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace boost;
namespace py = boost::python;
namespace np = boost::python::numpy;

typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef graph_traits<Graph>::vertex_iterator VertexIter;
typedef std::vector<int> IntVec;
typedef std::map<IntVec, int> VecIntMap;

enum class RemovalStrategy { random, targeted};

void print_vector(IntVec& vec_to_print) {
    for (IntVec::iterator it=vec_to_print.begin(); it!=vec_to_print.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << std::endl;
}

void populate_index_labels(Graph& g, IntVec& index_labels) {
    for (auto v : boost::make_iterator_range(vertices(g))) {
       index_labels[v] = v;
    }
}

void generate_random_sequence(Graph& g, IntVec& removal_sequence, int N) {
    populate_index_labels(g, removal_sequence);
    std::random_shuffle (removal_sequence.begin(), removal_sequence.end());
}

void generate_targeted_sequence(Graph& g, const IntVec& node_degrees, IntVec& removal_sequence, int N) {
    populate_index_labels(g, removal_sequence);

    IntVec random_sequence(N);
    generate_random_sequence(g, random_sequence, N);

    std::sort(
        removal_sequence.begin(),
        removal_sequence.end(),
        [node_degrees, random_sequence](int a, int b) {
            return (node_degrees[a] > node_degrees[b]) || (node_degrees[a] == node_degrees[b] && random_sequence[a] > random_sequence[b]);
        }
    );
}

void generate_removal_sequence(RemovalStrategy strat, Graph& g, const IntVec& node_degrees, IntVec& removal_sequence, int N) {
    switch(strat) {
        case RemovalStrategy::random: return generate_random_sequence(g,removal_sequence, N);
        case RemovalStrategy::targeted: return generate_targeted_sequence(g, node_degrees, removal_sequence, N);
    }
}

bool should_cache_ncc(RemovalStrategy strat) {
    switch(strat) {
        case RemovalStrategy::random: return false;
        case RemovalStrategy::targeted: return true;
    }
}

double compute_critical_fraction(RemovalStrategy strat, Graph& g, const IntVec& node_degrees, VecIntMap& ncc_cache, int N, unsigned base_graph_hash, int sim_num, int random_seed) {
    int crit_removed = 0;
    std::srand (unsigned ( (base_graph_hash * sim_num * random_seed) ) );
    IntVec component(N);
    IntVec removal_sequence(N);
    generate_removal_sequence(strat, g, node_degrees, removal_sequence, N);

    for(int i=0; i<N-1; ++i) {
        int node_to_remove = removal_sequence[i];

        clear_vertex(node_to_remove, g);
        int nodes_removed = i+1;

        int num = -1;
        if(should_cache_ncc(strat)) {
            IntVec seq_so_far = IntVec(removal_sequence.begin(), removal_sequence.begin() + i + 1);

            auto search = ncc_cache.find(seq_so_far);
            if (search != ncc_cache.end()) {
                num = search->second;
            } else {
                num = connected_components(g, &component[0]) - nodes_removed;
                ncc_cache.insert(std::make_pair(seq_so_far, num));
            }
        }
        else {
            num = connected_components(g, &component[0]) - nodes_removed;
        }

        if (num > 1) {
            crit_removed = nodes_removed;
            break;
        }
        if (i == N-2) {
            crit_removed = N;
            break;
        }
    }

    double p = ((double) crit_removed) / ((double) N);
    return p;
}

double exp_critical_fraction(RemovalStrategy strat, int N, int M, np::ndarray const& edge_list, int num_sims, unsigned base_graph_hash, int random_seed) {
    int* input_ptr = reinterpret_cast<int*>(edge_list.get_data());

    Graph g;
    int first_node, second_node;
    int input_size = M * 2;

    for (int i = 0; i < input_size-1; i+=2) {
        first_node = *(input_ptr + i);
        second_node = *(input_ptr + i + 1);
        add_edge(first_node, second_node, g);
    }

    IntVec node_degrees(N);
    for (auto v : boost::make_iterator_range(vertices(g))) {
        node_degrees[v] = out_degree(v, g);
    }

    double cf_sum = 0;

    VecIntMap ncc_cache;
    for (int sim_num = 0; sim_num < num_sims; ++sim_num) {
        Graph g_copy;
        copy_graph(g, g_copy);
        double cf = compute_critical_fraction(strat, g_copy, node_degrees, ncc_cache, N, base_graph_hash, sim_num, random_seed);
        cf_sum += cf;
    }

    double exp_cf = cf_sum / ((double) num_sims);
    return exp_cf;
}

double critical_fraction_random(int N, int M, np::ndarray const& edge_list, int num_sims, unsigned base_graph_hash, int random_seed)
{
    return exp_critical_fraction(RemovalStrategy::random, N, M, edge_list, num_sims, base_graph_hash, random_seed);
}

double critical_fraction_targeted(int N, int M, np::ndarray const& edge_list, int num_sims, unsigned base_graph_hash, int random_seed)
{
    return exp_critical_fraction(RemovalStrategy::targeted, N, M, edge_list, num_sims, base_graph_hash, random_seed);
}


BOOST_PYTHON_MODULE(objective_functions_ext)
{
    Py_Initialize();
    np::initialize();

    def("critical_fraction_random", critical_fraction_random);
    def("critical_fraction_targeted", critical_fraction_targeted);
}
