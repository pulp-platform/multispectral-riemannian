/*
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __TEST_FUNCTIONAL_DOT_PROD_H__
#define __TEST_FUNCTIONAL_DOT_PROD_H__

#include "stdint.h"
#include "stdbool.h"

void cluster_entry(void* arg);
bool do_bench_aa(rt_perf_t* perf, int events);


#endif //__TEST_FUNCTIONAL_DOT_PROD_H__
