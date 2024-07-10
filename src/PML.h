#include <cmath>

struct PML_Node
{
    float idx;    
};

class PML
{    
private:
    int n_PML_nodes_per_part;    
    float order;
    PML_Node *part1;
    PML_Node *part2;        
    void init();

public:    
    PML(int n_PML_nodes_per_part_, float order_)
    {
        n_PML_nodes_per_part = n_PML_nodes_per_part_;
        order = order_;        

        init();
    }    

    ~PML()
    {
        delete [] part1;
        delete [] part2;
    }
};

void PML::init()
{    
    part1 = new PML_Node[n_PML_nodes_per_part];    
    part2 = new PML_Node[n_PML_nodes_per_part];    
    //part2 = new PML_Node[n_PML_nodes_per_part];        

    for (int i=0; i<n_PML_nodes_per_part; i++)
    {
        PML_Node node;
        node.idx = i;
        float idx_to_the_order = 
        part1[i] = node;
    }
}