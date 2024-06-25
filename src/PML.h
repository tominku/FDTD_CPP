#include <cmath>

class PML
{    
private:
    int n_PML;
    double order;
    double *part1_positions_x;
    double *part2_positions_x;
    double *part1_positions_y;
    double *part2_positions_y;
    void compute_coeffs();

public:
    PML(int _n_PML)
    {
        n_PML = _n_PML;
        order = 3;
        part1_positions_x = new double[n_PML];
        part2_positions_x = new double[n_PML];
        part1_positions_y = new double[n_PML];
        part2_positions_y = new double[n_PML];

        compute_coeffs();
    }    
};

void PML::compute_coeffs()
{    
    for (int i=0; i<n_PML; i++)
        part1_positions_x[i] = pow(i, order);
    for (int i=0; i<n_PML; i++)
        part2_positions_x[i] = pow(i, order);           
    for (int i=0; i<n_PML; i++)
        part1_positions_y[i] = pow(i, order);
    for (int i=0; i<n_PML; i++)
        part2_positions_y[i] = pow(i, order);                   
}