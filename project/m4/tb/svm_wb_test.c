/*
 * SPDX-FileCopyrightText: 2024 Adam Handwerger
 * SPDX-License-Identifier: Apache-2.0
 *
 * svm_wb_test — Wishbone register read/write test for user_project_wrapper
 *
 * Wishbone map (base 0x30000000):
 *   +0x04  CONTROL     RW  [0]=start [3]=kern_ready
 *   +0x08  STATUS      RO  [0]=done  [1]=error [8:6]=class
 *   +0x0C  NUM_SAMPLES RW  [9:0]
 *   +0x10  NUM_SV[0]   RW  [7:0]
 *   +0x14  NUM_SV[1]   RW  [7:0]
 *   +0x18  NUM_SV[2]   RW  [7:0]
 *   +0x1C  NUM_SV[3]   RW  [7:0]
 *   +0x20  NUM_SV[4]   RW  [7:0]
 *
 * GPIO[31:16] are MGMT outputs (not driven by user project).
 * Testbench monitors checkbits = mprj_io[31:16]:
 *   0xBB90 = test started
 *   0xBB91 = all register checks passed
 *   0xBB00 = failure (written before $finish on error)
 */

#include <defs.h>
#include <stub.c>

#define SVM_BASE        0x30000000
#define REG_CONTROL     (*(volatile uint32_t*)(SVM_BASE + 0x04))
#define REG_STATUS      (*(volatile uint32_t*)(SVM_BASE + 0x08))
#define REG_NUM_SAMPLES (*(volatile uint32_t*)(SVM_BASE + 0x0C))
#define REG_NUM_SV0     (*(volatile uint32_t*)(SVM_BASE + 0x10))
#define REG_NUM_SV1     (*(volatile uint32_t*)(SVM_BASE + 0x14))
#define REG_NUM_SV2     (*(volatile uint32_t*)(SVM_BASE + 0x18))
#define REG_NUM_SV3     (*(volatile uint32_t*)(SVM_BASE + 0x1C))
#define REG_NUM_SV4     (*(volatile uint32_t*)(SVM_BASE + 0x20))

void main()
{
    /* Configure GPIO[31:16] as management outputs for test signalling */
    reg_mprj_io_31 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_30 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_29 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_28 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_27 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_26 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_25 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_24 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_23 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_22 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_21 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_20 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_19 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_18 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_17 = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_16 = GPIO_MODE_MGMT_STD_OUTPUT;

    reg_mprj_xfer = 1;
    while (reg_mprj_xfer == 1);

    reg_spi_enable = 1;
    reg_wb_enable  = 1;

    /* Signal: test starting */
    reg_mprj_datal = 0xBB900000;

    /* --- NUM_SAMPLES write/readback --- */
    REG_NUM_SAMPLES = 100;
    if ((REG_NUM_SAMPLES & 0x3FF) != 100) {
        reg_mprj_datal = 0xBB000000;
        return;
    }

    /* --- NUM_SV[0..4] write/readback --- */
    REG_NUM_SV0 = 30;
    REG_NUM_SV1 = 25;
    REG_NUM_SV2 = 40;
    REG_NUM_SV3 = 35;
    REG_NUM_SV4 = 20;

    if ((REG_NUM_SV0 & 0xFF) != 30) { reg_mprj_datal = 0xBB000000; return; }
    if ((REG_NUM_SV1 & 0xFF) != 25) { reg_mprj_datal = 0xBB000000; return; }
    if ((REG_NUM_SV2 & 0xFF) != 40) { reg_mprj_datal = 0xBB000000; return; }
    if ((REG_NUM_SV3 & 0xFF) != 35) { reg_mprj_datal = 0xBB000000; return; }
    if ((REG_NUM_SV4 & 0xFF) != 20) { reg_mprj_datal = 0xBB000000; return; }

    /* --- CONTROL readback (default = 8: kern_ready preset) --- */
    /* Just verify bit[0] (start) is 0 after reset */
    if (REG_CONTROL & 0x1) { reg_mprj_datal = 0xBB000000; return; }

    /* Signal: all checks passed */
    reg_mprj_datal = 0xBB910000;
}
