#ifdef __cplusplus
#define RESTRICT
#else
#define RESTRICT restrict
#endif

typedef struct sim_param_t {
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

    void compute_density(int n, float *rho, float *pos_x, float *pos_y, float h, float mass);

    void compute_accel(
        int n,
        float *rho,
        float *pos_x,
        float *pos_y,
        float *v_x,
        float *v_y,
        float *a_x,
        float *a_y,
        float mass,
        sim_param_t params);

    void leapfrog(sim_state_t* s, double dt);

#ifdef __cplusplus
}
#endif
