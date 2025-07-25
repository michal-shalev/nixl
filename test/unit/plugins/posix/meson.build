# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Using globally defined aio_dep and paio variables from the root meson.build
if posix_aio or rt_dep.found()
    # Get Abseil dependencies
    absl_log_dep = dependency('absl_log', required: true)

    nixl_posix_app = executable('nixl_posix_test', 'nixl_posix_test.cpp',
                                dependencies: [nixl_dep, nixl_infra, absl_log_dep],
                                include_directories: [nixl_inc_dirs, utils_inc_dirs],
                                install: true)

    # Register the test with the test suite
    test('posix_plugin_test', nixl_posix_app)
endif
