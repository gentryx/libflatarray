#ifdef __cplusplus
extern "C" {
#endif

    struct sim_param_t {
        float h; /* Particle size */
        float dt; /* Time step */
        float rho0; /* Reference density */
        float k; /* Bulk modulus */
        float mu; /* Viscosity */
        float g; /* Gravity strength */
    };

    void compute_density(
        int n,
        float *rho,
        float *pos_x,
        float *pos_y,
        float h,
        float mass);

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
        struct sim_param_t params);

    void leapfrog(
        int n,
        float *pos_x,
        float *pos_y,
        float *v_x,
        float *v_y,
        float *vh_x,
        float *vh_y,
        float *a_x,
        float *a_y,
        double dt);

    void reflect_bc(
        int n,
        float *pos_x,
        float *pos_y,
        float *v_x,
        float *v_y,
        float *vh_x,
        float *vh_y);

#ifdef __cplusplus
}
#endif
