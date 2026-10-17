/* Exact integer frontier routing; no floating-point arithmetic. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

int64_t frontier_indices(const int64_t *shape, int64_t ndim,
    const int64_t *lv, int64_t nl, const int64_t *rv, int64_t nr,
    const int64_t *neighbors, int64_t nn, const int64_t *physical,
    int64_t site, int64_t nsites, int64_t radix,
    int64_t *sources, int64_t *targets) {
    int64_t *values = malloc((size_t)nsites * sizeof(int64_t));
    if (!values) return -1;
    int64_t lc=1, rc=1, count=0;
    for (int64_t k=0;k<nl;k++) lc*=radix;
    for (int64_t k=0;k<nr;k++) rc*=radix;
    int64_t dl=shape[0], dp=shape[1], dr=shape[ndim-1];
    for (int64_t l=0;l<lc;l++) for (int64_t p=0;p<dp;p++)
    for (int64_t rr=0;rr<rc;rr++) {
        for (int64_t k=0;k<nsites;k++) values[k]=-1;
        int64_t x=l;
        for (int64_t k=nl;k-->0;) {values[lv[k]]=x%radix; x/=radix;}
        if (values[site]>=0 && values[site]!=physical[p]) continue;
        values[site]=physical[p];
        x=rr; int valid=1;
        for (int64_t k=nr;k-->0;) {
            int64_t v=x%radix; x/=radix;
            if (values[rv[k]]>=0 && values[rv[k]]!=v) {valid=0; break;}
            values[rv[k]]=v;
        }
        if (!valid) continue;
        for (int64_t k=0;k<nn;k++) if (values[neighbors[k]]<0) valid=0;
        if (!valid) continue;
        for (int64_t a=0;a<dl;a++) for (int64_t b=0;b<dr;b++) {
            if (sources) {
                int64_t src=a*dp+p;
                for (int64_t k=1;k<nn;k++) src=src*shape[k+1]+values[neighbors[k]];
                sources[count]=src*dr+b;
                targets[count]=((l*dl+a)*dp+p)*(rc*dr)+rr*dr+b;
            }
            count++;
        }
    }
    free(values);
    return count;
}
