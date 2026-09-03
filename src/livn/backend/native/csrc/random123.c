/* Random123 philox4x32-10 and the normal deviate, as NEURON's nrnran123 uses
 * them, so a seeded Gfluct3 stream is reproduced draw for draw.
 *
 * NEURON's stream policy: counter word 0 is the sequence number, words 1..3
 * carry (id3, id1, id2), the key is (global index = 0, 0). Draws come from
 * the four counter outputs in turn before the sequence advances.
 */
#include <math.h>

#include "internal.h"

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

static void philox4x32_10(const uint32_t ctr[4], const uint32_t key[2], uint32_t out[4]) {
    uint32_t c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3];
    uint32_t k0 = key[0], k1 = key[1];
    int r;
    for (r = 0; r < 10; ++r) {
        uint64_t p0 = (uint64_t) PHILOX_M0 * c0;
        uint64_t p1 = (uint64_t) PHILOX_M1 * c2;
        uint32_t hi0 = (uint32_t) (p0 >> 32), lo0 = (uint32_t) p0;
        uint32_t hi1 = (uint32_t) (p1 >> 32), lo1 = (uint32_t) p1;
        uint32_t n0 = hi1 ^ c1 ^ k0;
        uint32_t n1 = lo1;
        uint32_t n2 = hi0 ^ c3 ^ k1;
        uint32_t n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
        k0 += PHILOX_W0;
        k1 += PHILOX_W1;
    }
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
}

void r123_setseq(R123Stream* s, uint32_t seq, int which) {
    if (which < 0 || which > 3) {
        which = 0;
    }
    s->which = which;
    s->c[0] = seq;
    philox4x32_10(s->c, s->k, s->r);
}

void r123_seed(R123Stream* s, uint32_t id1, uint32_t id2, uint32_t id3) {
    s->c[0] = 0;
    s->c[1] = id3;
    s->c[2] = id1;
    s->c[3] = id2;
    s->k[0] = 0;
    s->k[1] = 0;
    r123_setseq(s, 0, 0);
}

static uint32_t r123_ipick(R123Stream* s) {
    uint32_t rval = s->r[s->which++];
    if (s->which > 3) {
        s->which = 0;
        s->c[0]++;
        philox4x32_10(s->c, s->k, s->r);
    }
    return rval;
}

static double r123_dblpick(R123Stream* s) {
    /* open (0, 1): ((double) u + 1) / (2^32 + 1) */
    return ((double) r123_ipick(s) + 1.0) * (1.0 / 4294967297.0);
}

double r123_normal(R123Stream* s) {
    double w, x, y, u1, u2;
    do {
        u1 = r123_dblpick(s);
        u2 = r123_dblpick(s);
        u1 = 2.0 * u1 - 1.0;
        u2 = 2.0 * u2 - 1.0;
        w = (u1 * u1) + (u2 * u2);
    } while (w > 1);
    y = sqrt((-2.0 * log(w)) / w);
    x = u1 * y;
    return x;
}

double rcsd_random123_normal(unsigned id1, unsigned id2, unsigned id3, unsigned seq,
                             int index) {
    R123Stream s;
    double x = 0.0;
    int i;
    r123_seed(&s, id1, id2, id3);
    r123_setseq(&s, seq, 0);
    for (i = 0; i <= index; ++i) {
        x = r123_normal(&s);
    }
    return x;
}
