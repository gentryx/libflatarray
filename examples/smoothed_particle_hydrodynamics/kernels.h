#ifdef __cplusplus
extern "C" {
#endif

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
        float g,
        float h,
        float k,
        float rho0,
        float mu);

    void leapfrog(
        int n,
        float *pos_x,
        float *pos_y,
        float *v_x,
        float *v_y,
        float *a_x,
        float *a_y,
        double dt);

    void reflect_bc(
        int n,
        float *pos_x,
        float *pos_y,
        float *v_x,
        float *v_y);

#ifdef __cplusplus
}
#endif
