#ifdef __cplusplus
#define RESTRICT
#else
#define RESTRICT restrict
#endif

typedef struct sim_param_t {
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
    float* RESTRICT rho; /* Densities */
    float* RESTRICT pos_x; /* Positions */
    float* RESTRICT pos_y; /* Positions */
    float* RESTRICT vh_x; /* Velocities (half step) */
    float* RESTRICT vh_y; /* Velocities (half step) */
    float* RESTRICT v_x; /* Velocities (full step) */
    float* RESTRICT v_y; /* Velocities (full step) */
    float* RESTRICT a_x; /* Acceleration */
    float* RESTRICT a_y; /* Acceleration */
} sim_state_t;

#ifdef __cplusplus
extern "C" {
#endif

    void write_frame_data(int cycle, int n, float* pos_x, float* pos_y);
    void compute_accel(sim_state_t* state, sim_param_t params);
    void leapfrog_start(sim_state_t* s, double dt);
    void leapfrog_step(sim_state_t* s, double dt);
    void compute_density(sim_state_t* s, float h);

#ifdef __cplusplus
}
#endif
