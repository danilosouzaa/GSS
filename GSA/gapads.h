/**
 * @file   gapads.h
 *
 * @brief  Auxiliary Data Structure (ADS)
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#ifndef __gapads_h
#define __gapads_h

#include "gpulib/gpu.h"
#include "gpulib/types.h"

// ################################################################################
// //
// ## ## //
// ##                               CONSTANTS & MACROS ## //
// ## ## //
// ################################################################################
// //

#define COST_INFTY 0x7fffffff

#define GAPMI_OROPT(k) int(GAPMI_OROPT1 + k - 1)
#define GAPMI_OROPT_K(id) (int(id) - 1)
#define GAPMI_KERNEL(m) int((m)&0xf)

// ################################################################################
// //
// ## ## //
// ##                               CONSTANTS & MACROS ## //
// ## ## //
// ################################################################################
// //

/*
 * ADS buffer
 *
 *     info      coords[x,y]        [W|Id]              T               C[0]
 * C[size-1]
 * +---------+----------------+----------------+----------------+----------------+-----+----------------+
 * | b elems | size elems |gap| size elems |gap| size elems |gap| size elems
 * |gap| ... | size elems |gap|
 * +---------+----------------+----------------+----------------+----------------+-----+----------------+
 *
 * b = GAPP_ADSINFO_ELEMS = GPU_GLOBAL_ALIGN/sizeof(uint) = 128/4 = 32 elems
 *
 * info[0]  : size (ADS buffer size in bytes)
 * info[1]  : rowElems
 * info[2]  : solElems (sol size)
 * info[3]  : solCost
 * info[4]  : tour
 * info[5]  : round
 * info[6]  : reserved
 * ...
 * info[b-1]: reserved
 */

#define ADS_INFO_SIZE GPU_GLOBAL_ALIGN
#define ADS_INFO_ELEMS (ADS_INFO_SIZE / sizeof(uint))

// #define ADS_INFO_ADS_SIZE       0
// #define ADS_INFO_ROW_ELEMS      1
// #define ADS_INFO_SOL_ELEMS      2
// #define ADS_INFO_SOL_COST       3
// #define ADS_INFO_TOUR           4
// #define ADS_INFO_ROUND          5

#define ADS_COORD_PTR(ads) (puint(ads + 1))
#define ADS_SOLUTION_PTR(ads, r) (puint(ads + 1) + r)
#define ADS_TIME_PTR(ads, r) (puint(ads + 1) + 2 * r)
#define ADS_COST_PTR(ads, r) (puint(ads + 1) + 3 * r)

// #################################################################################
// //
// ## ## //
// ##                                 DATA TYPES ## //
// ## ## //
// #################################################################################
// //

/*!
 * GAPArch
 */
typedef enum {
  GAPSA_CPU,   ///< CPU
  GAPSA_SGPU,  ///< Single GPU
  GAPSA_MGPU,  ///< Multi  GPU
} GAPArch;

/*!
 * GAPHeuristic
 */
typedef enum {
  GAPSH_RVND,  ///< RVND
  GAPSH_DVND,  ///< DVND
} GAPHeuristic;

/*!
 * GAPInprovement
 */
typedef enum {
  GAPSI_BEST,   ///< Best improvement
  GAPSI_MULTI,  ///< Multi improvement
} GAPImprovement;

/*!
 * GAPHistorySelect
 */
typedef enum {
  GAPSS_BEST,  ///< Best solution
  GAPSS_RAND,  ///< Random solution
} GAPHistorySelect;

/*!
 * GAPADSInfo
 */
struct GAPADSInfo {
  uint size;      ///< ADS buffer size in bytes
  uint rowElems;  ///< ADS row  elems
  uint solElems;  ///< Solution elems
  uint solCost;   ///< Solution cost
  uint tour;      ///< Tour (0/1)
  uint round;     ///< Round: 0=0.0, 1=0.5
};

/*!
 * GAPADSData
 */
// 128 Bytes
union GAPADSData {
  GAPADSInfo s;
  uint32_t v[ADS_INFO_ELEMS];  // 128 / 4
};

static_assert(sizeof(GAPADSInfo) == 24);
static_assert(sizeof(GAPADSData) == 4 * 32);

/*!
 * GAPKernelId
 */
typedef enum {
  GAPMI_SWAP,
  GAPMI_2OPT,
  GAPMI_OROPT1,
  GAPMI_OROPT2,
  GAPMI_OROPT3,
  GAPMI_OROPT4,
  GAPMI_OROPT5,
  GAPMI_OROPT6,
  GAPMI_OROPT7,
  GAPMI_OROPT8,
  GAPMI_OROPT9,
  GAPMI_OROPT10,
  GAPMI_OROPT11,
  GAPMI_OROPT12,
  GAPMI_OROPT13,
  GAPMI_OROPT14,
  GAPMI_OROPT15,
  GAPMI_OROPT16,
  GAPMI_OROPT17,
  GAPMI_OROPT18,
  GAPMI_OROPT19,
  GAPMI_OROPT20,
  GAPMI_OROPT21,
  GAPMI_OROPT22,
  GAPMI_OROPT23,
  GAPMI_OROPT24,
  GAPMI_OROPT25,
  GAPMI_OROPT26,
  GAPMI_OROPT27,
  GAPMI_OROPT28,
  GAPMI_OROPT29,
} GAPMoveId;

/*!
 * GAPMove64
 */


struct GAPMove64 {
  unsigned int id : 4;
  unsigned int i : 14;
  unsigned int j : 14;
  int cost;  // REMOVED :32 !! Goddamn Eyder...
};

/*!
 * GAPMovePack
 */
union GAPMovePack {
  GAPMove64 s;
  ulong w;
  int i[2];
  uint u[2];
  long l[1];
};

/*!
 * GAPMove
 */
struct GAPMove {
  GAPMoveId id;
  int i;
  int j;
  int cost;
};

/*!
 * GAPPointer
 */
union GAPPointer {
  const void* p_cvoid;
  void* p_void;
  char* p_char;
  byte* p_byte;
  short* p_short;
  ushort* p_ushort;
  int* p_int;
  uint* p_uint;
  long* p_long;
  ulong* p_ulong;
  llong* p_llong;
  ullong* p_ullong;
  GAPMove* p_move;
  GAPMove64* p_move64;
  GAPMovePack* p_mpack;
};

// #################################################################################
// //
// ## ## //
// ##                              GLOBAL VARIABLES ## //
// ## ## //
// #################################################################################
// //

/*!
 * Architecture names
 */
extern const char* nameArch[];

extern const char* nameLongArch[];

/*!
 * Heuristics names
 */
extern const char* nameHeuristic[];

/*!
 * Movement names
 */
extern const char* nameMove[];

/*!
 * Movement merge names
 */
extern const char* nameImprov[];

extern const char* nameLongImprov[];

/*!
 * Clear movement
 */
extern const GAPMove64 MOVE64_NONE, MOVE64_INFTY;

extern const GAPMove MOVE_NONE, MOVE_INFTY;

// #################################################################################
// //
// ## ## //
// ##                                 FUNCTIONS ## //
// ## ## //
// #################################################################################
// //

inline void move64ToMove(GAPMove& m1, GAPMove64& m2) {
  m1.id = GAPMoveId(m2.id);
  m1.i = m2.i;
  m1.j = m2.j;
  m1.cost = m2.cost;
}

inline void moveToMove64(GAPMove64& m1, GAPMove& m2) {
  m1.id = m2.id;
  m1.i = m2.i;
  m1.j = m2.j;
  m1.cost = m2.cost;
}

#endif
