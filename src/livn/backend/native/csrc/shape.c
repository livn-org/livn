/* NEURON's 3-D shape of a stylised section tree, and the segment areas and
 * axial resistances it implies.
 *
 * The NEURON backend calls h.define_shape() once its cells exist, which
 * gives every section a list of 3-D points whose coordinates and diameters
 * are single-precision floats (struct Pt3d in nrnoc/section.h). From then on
 * nrn_area_ri integrates area and resistance from those points
 * (diam_from_list), so a diameter that is not representable in single
 * precision is off by up to 6e-8 relative to the cylinder formula, and a
 * cell whose axon leaves the soma at float(pi) picks up a 1e-6 um lateral
 * offset. Both are far below anything physiological, but a difference of
 * 1e-8 in a conductance is amplified a million-fold on the upstroke of a
 * spike, and a step-for-step contract with NEURON has to carry them.
 *
 * This file is nrn_define_shape, stor_pt3d/nrn_pt3dmodified,
 * nrn_length_change, nrn_diam_change and diam_from_list from
 * nrnoc/treeset.cpp (NEURON 9.0.1), with the same float and double
 * variables in the same expressions. The order in which a parent's children
 * are visited is the order of NEURON's sibling list (cabcode.cpp,
 * nrn_add_sibling_list): by connection position, later-created first among
 * equals. Root sections sit at z = 100 * (section index); the value is
 * irrelevant, every point of a tree shares it.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* nrn_pt3dmodified: arc length from the float coordinates, from point i0 on */
static void pt3d_modified(Section* sec, int i0) {
    int i, n = sec->npt3d;
    if (i0 == 0) {
        sec->pt3d[0].arc = 0.0;
        i0 = 1;
    }
    for (i = i0; i < n; ++i) {
        Pt3d* p = sec->pt3d + i - 1;
        /* NEURON subtracts the float fields into doubles; the subtraction
         * itself is single precision */
        float fx = sec->pt3d[i].x - p->x;
        float fy = sec->pt3d[i].y - p->y;
        float fz = sec->pt3d[i].z - p->z;
        double t1 = fx, t2 = fy, t3 = fz;
        sec->pt3d[i].arc = p->arc + sqrt(t1 * t1 + t2 * t2 + t3 * t3);
    }
    sec->L = sec->pt3d[n - 1].arc;
}

static int stor_pt3d(Section* sec, double x, double y, double z, double d) {
    int n = sec->npt3d;
    if (n + 1 > sec->pt3d_cap) {
        int cap = sec->pt3d_cap ? sec->pt3d_cap * 2 : 8;
        Pt3d* grown;
        while (cap < n + 1) {
            cap *= 2;
        }
        grown = (Pt3d*) realloc(sec->pt3d, (size_t) cap * sizeof(Pt3d));
        if (grown == NULL) {
            rcsd_set_error("out of memory");
            return RCSD_ERROR;
        }
        sec->pt3d = grown;
        sec->pt3d_cap = cap;
    }
    sec->npt3d = n + 1;
    sec->pt3d[n].x = (float) x;
    sec->pt3d[n].y = (float) y;
    sec->pt3d[n].z = (float) z;
    sec->pt3d[n].d = (float) d;
    pt3d_modified(sec, n);
    return RCSD_OK;
}

void rcsd_shape_free(Section* sec) {
    free(sec->pt3d);
    sec->pt3d = NULL;
    sec->npt3d = 0;
    sec->pt3d_cap = 0;
}

/* section_length: with points, L is the arc of the last one */
static double section_length(Section* sec) {
    double x;
    if (sec->npt3d) {
        sec->L = sec->pt3d[sec->npt3d - 1].arc;
    }
    x = sec->L;
    if (x <= 1e-9) {
        x = 1e-9;
        sec->L = x;
    }
    return x;
}

/* the sibling list of `parent`: children in NEURON's order, into `order`;
 * returns the count */
static int sibling_order(RCSDSim* sim, int parent, int* order) {
    int n = 0;
    size_t s;
    for (s = 0; s < sim->sections.n; ++s) {
        Section* sec = &sim->sections.data[s];
        double x;
        int k, at;
        if (sec->parent_section != parent) {
            continue;
        }
        /* nrn_add_sibling_list: before the first sibling at or beyond x */
        x = sec->parent_x;
        at = n;
        for (k = 0; k < n; ++k) {
            if (x <= sim->sections.data[order[k]].parent_x) {
                at = k;
                break;
            }
        }
        for (k = n; k > at; --k) {
            order[k] = order[k - 1];
        }
        order[at] = (int) s;
        ++n;
    }
    return n;
}

/* nrn_define_shape for the sections that have no points yet */
int rcsd_shape_define(RCSDSim* sim) {
    size_t i;
    int* order = NULL;
    size_t order_cap = 0;
    for (i = 0; i < sim->sections.n; ++i) {
        Section* sec = &sim->sections.data[i];
        Section* psec = sec->parent_section >= 0 ? &sim->sections.data[sec->parent_section]
                                                 : NULL;
        float x, y, z, dz, x1, y1;
        float nch, ich = 0.0f, angle;
        double arc, len;
        int j;
        if (sec->npt3d) {
            continue;
        }
        dz = 100.0f;
        arc = psec ? sec->parent_x : 0.0;
        if (psec == NULL) {
            x = 0.0f;
            y = 0.0f;
            z = (float) i * dz;
            x1 = 1.0f;
            y1 = 0.0f;
        } else {
            double arc1 = arc;
            const Pt3d* first = &psec->pt3d[0];
            const Pt3d* last = &psec->pt3d[psec->npt3d - 1];
            x1 = last->x - first->x;
            y1 = last->y - first->y;
            /* arc0at0(psec): every section here is joined by its 0 end */
            if (arc1 < 0.5) {
                x1 = -x1;
                y1 = -y1;
            }
            x = (float) (last->x * arc1 + first->x * (1 - arc1));
            y = (float) (last->y * arc1 + first->y * (1 - arc1));
            z = (float) (last->z * arc1 + first->z * (1 - arc1));
        }
        if (fabs((double) y1) < 1e-6 && fabs((double) x1) < 1e-6) {
            angle = 0.0f;
        } else {
            angle = (float) atan2((double) y1, (double) x1);
        }
        if (arc > 0.0 && arc < 1.0) {
            angle = (float) (angle + 3.14159265358979323846 / 2.0);
        }
        nch = 0.0f;
        if (psec) {
            int n, k;
            if (order_cap < sim->sections.n) {
                int* grown = (int*) realloc(order, sim->sections.n * sizeof(int));
                if (grown == NULL) {
                    free(order);
                    rcsd_set_error("out of memory");
                    return RCSD_ERROR;
                }
                order = grown;
                order_cap = sim->sections.n;
            }
            n = sibling_order(sim, sec->parent_section, order);
            for (k = 0; k < n; ++k) {
                if ((size_t) order[k] == i) {
                    ich = nch;
                }
                if (arc == sim->sections.data[order[k]].parent_x) {
                    nch = nch + 1.0f;
                }
            }
        }
        if (nch > 1.0f) {
            angle = (float) (angle + ich / (nch - 1.0) * 0.8 - 0.4);
        }
        len = section_length(sec);
        x1 = (float) (x + len * cos((double) angle));
        y1 = (float) (y + len * sin((double) angle));
        if (stor_pt3d(sec, x, y, z, sec->diam) != RCSD_OK) {
            free(order);
            return RCSD_ERROR;
        }
        for (j = 0; j < sec->nseg; ++j) {
            double frac = ((double) j + 0.5) / (double) sec->nseg;
            if (stor_pt3d(sec, x * (1 - frac) + x1 * frac, y * (1 - frac) + y1 * frac, z,
                          sec->diam) != RCSD_OK) {
                free(order);
                return RCSD_ERROR;
            }
        }
        if (stor_pt3d(sec, x1, y1, z, sec->diam) != RCSD_OK) {
            free(order);
            return RCSD_ERROR;
        }
        /* don't let above change length due to round-off errors */
        sec->pt3d[sec->npt3d - 1].arc = len;
        sec->L = len;
    }
    free(order);
    return RCSD_OK;
}

/* `sec.L = d` once the shape exists: nrn_length_change */
void rcsd_shape_length_change(Section* sec, double d) {
    int i;
    double x0, y0, z0, fac, l;
    sec->L = d;
    if (sec->npt3d == 0) {
        return;
    }
    x0 = sec->pt3d[0].x;
    y0 = sec->pt3d[0].y;
    z0 = sec->pt3d[0].z;
    l = sec->pt3d[sec->npt3d - 1].arc;
    fac = d / l;
    for (i = 0; i < sec->npt3d; ++i) {
        sec->pt3d[i].arc = sec->pt3d[i].arc * fac;
        sec->pt3d[i].x = (float) (x0 + (sec->pt3d[i].x - x0) * fac);
        sec->pt3d[i].y = (float) (y0 + (sec->pt3d[i].y - y0) * fac);
        sec->pt3d[i].z = (float) (z0 + (sec->pt3d[i].z - z0) * fac);
    }
    sec->recalc_area = 1;
}

/* `sec.diam = d` once the shape exists: nrn_diam_change */
void rcsd_shape_diam_change(Section* sec, double d) {
    int i;
    double L;
    sec->diam = d;
    if (sec->npt3d == 0) {
        return;
    }
    L = section_length(sec);
    if (fabs(L - sec->pt3d[sec->npt3d - 1].arc) > 0.001) {
        rcsd_shape_length_change(sec, L);
    }
    for (i = 0; i < sec->npt3d; ++i) {
        sec->pt3d[i].d = (float) d;
    }
    sec->recalc_area = 1;
}

/* diam_from_list: trapezoidal integration of diameter, area and resistance
 * over segment `inode` from the points; fills the node's area and rinv and
 * returns the right-half resistance (MOhm). `it` carries what NEURON keeps
 * in static variables between the calls for consecutive segments. */
typedef struct {
    int j;
    double x1, y1, ds;
} DiamIter;

static double diam_from_list(RCSDSim* sim, Section* sec, int inode, double rparent,
                             DiamIter* it) {
    int ihalf;
    double si, sip;
    double diam, delta, temp, ri, area, ra, rleft = 0.0;
    int npt;
    int node = sec->node0 + inode;

    if (inode == 0) {
        it->j = 0;
        it->x1 = sec->pt3d[0].arc;
        it->y1 = fabs((double) sec->pt3d[0].d);
        it->ds = sec->pt3d[sec->npt3d - 1].arc / ((double) sec->nseg);
    }
    si = (double) inode * it->ds;
    npt = sec->npt3d;
    diam = 0.0;
    area = 0.0;
    ra = sec->Ra;
    for (ihalf = 0; ihalf < 2; ihalf++) {
        ri = 0.0;
        sip = si + it->ds / 2.0;
        for (;;) {
            int jp, jnext;
            double x2, y2, xj, xjp;
            jp = it->j + 1;
            xj = sec->pt3d[it->j].arc;
            xjp = sec->pt3d[jp].arc;
            if (xjp > sip || jp == npt - 1) {
                double frac;
                if (fabs(xjp - xj) < 1e-10) {
                    frac = 1;
                } else {
                    frac = (sip - xj) / (xjp - xj);
                }
                x2 = sip;
                y2 = (1.0 - frac) * fabs((double) sec->pt3d[it->j].d) +
                     frac * fabs((double) sec->pt3d[jp].d);
                jnext = it->j;
            } else {
                x2 = xjp;
                y2 = fabs((double) sec->pt3d[jp].d);
                jnext = jp;
            }
            delta = (x2 - it->x1);
            diam += (y2 + it->y1) * delta;
            if (delta < 1e-15) {
                delta = 1e-15;
            }
            if ((temp = y2 * it->y1 / delta) == 0) {
                temp = 1e-15;
            }
            ri += 1 / temp;
            temp = 0.5 * (y2 - it->y1);
            temp = sqrt(delta * delta + temp * temp);
            area += (y2 + it->y1) * temp;
            it->x1 = x2;
            it->y1 = y2;
            if (it->j == jnext) {
                break;
            }
            it->j = jnext;
        }
        if (ihalf == 0) {
            rleft = ri * ra / M_PI * (4.0 * 0.01);
        } else {
            ri = ri * ra / M_PI * (4.0 * 0.01);
        }
        si = sip;
    }
    sim->rinv[node] = 1.0 / (rparent + rleft);
    diam *= 0.5 / it->ds;
    if (inode == 0 && (fabs(diam - sec->diam) > 1e-9 || diam < 1e-5)) {
        /* NEURON updates the segment's diam parameter; the first segment's
         * value stands for the section's */
        sec->diam = diam;
    }
    sim->area[node] = area * 0.5 * M_PI;
    return ri;
}

/* nrn_area_ri for one section */
void rcsd_shape_area_ri(RCSDSim* sim, Section* sec) {
    double ra, dx, rright, rleft;
    int j;
    DiamIter it;
    if (sec->npt3d) {
        sec->L = sec->pt3d[sec->npt3d - 1].arc;
    }
    ra = sec->Ra;
    dx = section_length(sec) / ((double) sec->nseg);
    rright = 0.0;
    for (j = 0; j < sec->nseg; ++j) {
        int node = sec->node0 + j;
        if (sec->npt3d > 1) {
            rright = diam_from_list(sim, sec, j, rright, &it);
        } else {
            double diam = sec->diam;
            sim->area[node] = M_PI * diam * dx;
            rleft = 1e-2 * ra * (dx / 2) / (M_PI * diam * diam / 4.0);
            sim->rinv[node] = 1.0 / (rleft + rright);
            rright = rleft;
        }
    }
    sim->area[sec->end_node] = 1e2;
    sim->rinv[sec->end_node] = 1.0 / rright;
    sec->recalc_area = 0;
}
