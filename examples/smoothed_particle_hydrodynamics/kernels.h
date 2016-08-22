typedef struct sim_param_t {
    char* fname; /* File name */
    int nframes; /* Number of frames */
    int npframe; /* Steps per frame */
    float h; /* Particle size */
    float dt; /* Time step */
    float rho0; /* Reference density */
    float k; /* Bulk modulus */
    float mu; /* Viscosity */
    float g; /* Gravity strength */
} sim_param_t;
int get_params(int argc, char** argv, sim_param_t* params);

typedef struct sim_state_t {
    int n; /* Number of particles */
    float mass; /* Particle mass */
    float* restrict rho; /* Densities */
    /* fixme: use x,y,z */
    float* restrict pos_x; /* Positions */
    float* restrict pos_y; /* Positions */
    float* restrict vh_x; /* Velocities (half step) */
    float* restrict vh_y; /* Velocities (half step) */
    float* restrict v_x; /* Velocities (full step) */
    float* restrict v_y; /* Velocities (full step) */
    float* restrict a_x; /* Acceleration */
    float* restrict a_y; /* Acceleration */
} sim_state_t;

