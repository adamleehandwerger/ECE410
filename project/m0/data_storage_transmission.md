# Anomaly Detection: Data Storage & Phone Transmission

## Question
**When an anomaly appears, can the sensor data be stored and transmitted to a phone?**

## Answer: Yes - Highly Practical Feature!

Storing and transmitting anomaly data to a phone is not only feasible but highly recommended for clinical validation, patient monitoring, and doctor review. This is a core feature for real-world cardiac monitoring wearables.

---

## System Architecture

### End-to-End Workflow
```
ECG Sensor (360 Hz)
        ↓
MCU Feature Extraction (256 features)
        ↓
Chiplet SVM Classification
        ↓
Anomaly Detected! (PVC/AFib/VT/SVT)
        ↓
Store: Raw ECG + Features + Classification
        ↓
Bluetooth LE Transmission
        ↓
Phone App (iOS/Android)
        ↓
Cloud Sync (Optional)
        ↓
Doctor Dashboard / EMR Integration
```

---

## What to Store When Anomaly Detected

### Minimal Package (~10 KB per event)

**Recommended for battery-constrained devices:**

| Data Type | Size | Description |
|-----------|------|-------------|
| **Raw ECG segment** | 7.2 KB | 10 seconds @ 360 Hz × 2 bytes |
| **Extracted features** | 1 KB | 256 features × 4 bytes |
| **Classification** | 24 bytes | Predicted class + 5 confidence scores |
| **Metadata** | 100 bytes | Timestamp, HR, battery, position |
| **Total** | ~8.3 KB | Per anomaly event |

**Storage capacity:** 2 MB flash → 240 events

### Extended Package (~50 KB per event)

**For diagnostic/research mode:**

| Data Type | Size | Description |
|-----------|------|-------------|
| **Extended ECG** | 21.6 KB | 30 seconds context @ 360 Hz |
| **R-R intervals** | 400 bytes | Last 100 heartbeats |
| **Trend data** | 2 KB | HRV, activity, temperature |
| **Device state** | 1 KB | Motion, position, artifacts |
| **Total** | ~25 KB | Enhanced diagnostic data |

**Storage capacity:** 2 MB flash → 80 events

### Custom Package (Configurable)

**User/doctor can configure:**
- ECG window duration (5-60 seconds)
- Pre-event buffer (capture 5s before anomaly)
- Post-event buffer (capture 5s after anomaly)
- Include accelerometer data (motion artifacts)
- Include temperature/SpO2 if available

---

## Storage Options

### Option 1: MCU Internal Flash (Recommended)

**Typical MCU Flash:** 2-4 MB

**Capacity:**
- Minimal package (8.3 KB): **240-480 events**
- Extended package (25 KB): **80-160 events**

**Implementation:**
- Circular buffer (oldest overwritten when full)
- Flash wear leveling (distribute writes)
- Persist across power cycles
- Low power overhead

**Pros:**
- ✅ No external components
- ✅ Low power
- ✅ Fast access
- ✅ Survives reboot

**Cons:**
- ❌ Limited capacity
- ❌ Flash wear over years

### Option 2: External Flash/SD Card

**128 MB - 1 GB external storage**

**Capacity:**
- Minimal package: **15,000 - 120,000 events**
- Extended package: **5,000 - 40,000 events**
- Months to years of data

**Implementation:**
- SPI/SDIO interface to MCU
- Wear leveling file system (LittleFS, FatFS)
- Removable SD card option

**Pros:**
- ✅ Massive storage
- ✅ Removable (doctor can read card)
- ✅ Long-term logging

**Cons:**
- ❌ Extra component cost (~$1-5)
- ❌ Higher power (1-10 mA when active)
- ❌ Mechanical reliability (SD card)

### Hybrid Approach (Best Practice)

```
Recent Events (last 100):
└─ MCU internal flash (fast, always available)

Historical Events (older):
└─ External flash (archived, bulk analysis)

Critical Events (VT/VFib):
└─ Both (redundancy for safety)
```

---

## Bluetooth LE Transmission

### BLE Architecture

**Hardware:**
- Nordic nRF52 series (BLE 5.0+)
- ESP32 (BLE + WiFi)
- STM32WB (BLE + Cortex-M4)
- TI CC2640 (low-power BLE)

**Protocol:**
- BLE GATT (Generic Attribute Profile)
- Custom service UUID for cardiac data
- Characteristics for ECG, events, status

### Transmission Modes

#### Mode 1: Real-Time Critical Alert

**Use Case:** Life-threatening arrhythmias (VT, VFib)

```
Anomaly Detected → Immediate BLE Alert → Phone Notification
```

**Characteristics:**
- **Latency:** <1 second from detection to phone
- **Power:** Brief burst (15 mA for 50-200 ms)
- **Reliability:** Retry until acknowledged
- **User experience:** Phone vibrates/rings immediately

**Implementation:**
```c
void onCriticalAnomaly() {
    storeEvent(ECG_buffer, features, VT);
    ble_sendPriorityAlert(VT, confidence);
    soundBuzzer();  // Optional haptic/audio alert
    waitForAck(timeout_5s);
}
```

#### Mode 2: Batch Upload (Non-Critical)

**Use Case:** PVC, AFib, routine monitoring

```
Store Anomalies → Periodic Sync → Upload When Phone Nearby
```

**Characteristics:**
- **Frequency:** Every 1-4 hours or when connected
- **Power:** Lower (batch transmission more efficient)
- **Data:** Multiple events per connection
- **User experience:** Background sync, no interruption

**Implementation:**
```c
void scheduleBatchUpload() {
    if (pending_events >= 10 || time_since_sync > 4_hours) {
        if (ble_isConnected()) {
            uploadPendingEvents();
        } else {
            ble_advertise();  // Request phone connection
        }
    }
}
```

#### Mode 3: Continuous Streaming (Diagnostic)

**Use Case:** Doctor visit, calibration, troubleshooting

```
Stream Live ECG → Phone App → Real-Time Visualization
```

**Characteristics:**
- **Data rate:** 360 samples/sec × 2 bytes = 720 bytes/sec
- **Latency:** 10-50 ms (real-time)
- **Power:** High (10-20 mA continuous)
- **Duration:** 5-30 minutes (not for 24/7 use)

**Use Cases:**
- Doctor examining patient
- Initial device setup
- Debugging/validation
- Event playback with live ECG

---

## Phone App Features

### Real-Time Monitoring View

**Display Elements:**
- Live ECG waveform (scrolling)
- Current heart rate (BPM)
- Rhythm classification (Normal/PVC/AFib/VT/SVT)
- Battery level, signal quality
- Connection status

### Alert Notifications

**Critical Alerts (VT/VFib):**
- Full-screen alert (cannot dismiss easily)
- Loud sound + vibration
- Optional: Auto-call emergency contact
- Optional: Send GPS location to caregiver

**Non-Critical Alerts (PVC/AFib):**
- Banner notification
- Badge count on app icon
- Summary: "3 AFib episodes detected today"

### Historical Data Browser

**Features:**
- Calendar view of all events
- Filter by type (Normal/PVC/AFib/VT/SVT)
- Event list with timestamps
- Tap to view full ECG waveform
- Zoom/pan ECG viewer
- Playback speed control

### Trend Analysis

**Visualizations:**
- AFib burden (% of time in AFib)
- PVC frequency (events per hour)
- Heart rate variability trends
- Circadian rhythm analysis
- Activity correlation

### Data Export

**Export Formats:**
- PDF report for doctor
- CSV for spreadsheet analysis
- HL7/FHIR for EMR integration
- WFDB format for research

**Sharing:**
- Email to doctor
- AirDrop/Share sheet
- Upload to health apps (Apple Health, Google Fit)
- Print physical report

### Cloud Sync

**Features:**
- Automatic backup
- Multi-device access (phone + tablet)
- Doctor dashboard access
- Family/caregiver view
- Automated alerts to caregivers

---

## Power Consumption Analysis

### BLE Power Budget

**Scenario 1: Alert-Only Mode**
```
Assumptions:
- 1 anomaly per hour (average)
- 10 KB transmission @ 1 Mbps BLE
- 15 mA during transmission
- Transmission time: 80 ms

Calculation:
- Power per event: 15 mA × 0.08s = 1.2 mAh
- Daily: 1.2 mAh × 24 events = 28.8 mAh
```

**Scenario 2: Batch Sync (4× daily)**
```
Assumptions:
- 6 anomalies per sync
- 60 KB total transmission
- 15 mA during transmission
- Transmission time: 480 ms

Calculation:
- Power per sync: 15 mA × 0.48s = 7.2 mAh
- Daily: 7.2 mAh × 4 = 28.8 mAh
```

**Scenario 3: Continuous Advertising**
```
Assumptions:
- BLE advertising every 1 second
- 5 mA average when advertising
- 50 ms per advertisement

Calculation:
- Daily: 5 mA × 0.05s × 86400 = 21.6 mAh
```

### Total System Power Budget

| Component | Power (mAh/day) | Percentage |
|-----------|-----------------|------------|
| **MCU (feature extraction)** | 10-15 mAh | 20-30% |
| **Chiplet (SVM inference)** | 0.6 mAh | 1% |
| **BLE (alerts + sync)** | 30 mAh | 50-60% |
| **ECG sensor** | 5-10 mAh | 10-20% |
| **Total** | **45-55 mAh/day** | 100% |

**Battery Life:**
- 500 mAh battery: **9-11 days**
- 1000 mAh battery: **18-22 days**

### Power Optimization Strategies

**1. Adaptive Advertising**
- Advertise aggressively when anomaly detected
- Reduce advertising when no events (save power)
- Stop advertising when phone connected

**2. Connection Interval Optimization**
- Fast interval (7.5 ms) for critical alerts
- Slow interval (100 ms) for batch sync
- Balance latency vs power

**3. Data Compression**
- Compress ECG with delta encoding (50% reduction)
- Compress features with quantization
- Trade: CPU for BLE power savings

---

## Implementation Example

### MCU Firmware Logic

```c
// Event storage structure
typedef struct {
    uint32_t timestamp;
    uint8_t classification;  // 0=N, 1=PVC, 2=AFib, 3=VT, 4=SVT
    float confidence[5];
    int16_t ecg_segment[3600];  // 10s @ 360 Hz
    float features[256];
    uint8_t heart_rate;
    uint8_t battery_level;
} AnomalyEvent_t;

// Circular buffer in flash
#define MAX_EVENTS 240
AnomalyEvent_t event_buffer[MAX_EVENTS];
uint16_t event_head = 0;
uint16_t event_count = 0;

void onAnomalyDetected(uint8_t predicted_class, float* confidence) {
    // Create event record
    AnomalyEvent_t event;
    event.timestamp = getRTC_timestamp();
    event.classification = predicted_class;
    memcpy(event.confidence, confidence, 5 * sizeof(float));
    memcpy(event.ecg_segment, ecg_buffer, 3600 * sizeof(int16_t));
    memcpy(event.features, feature_vector, 256 * sizeof(float));
    event.heart_rate = getCurrentHR();
    event.battery_level = getBatteryLevel();
    
    // Store in circular buffer
    storeEventToFlash(&event);
    
    // Handle based on severity
    if (predicted_class == VT || predicted_class == VFIB) {
        // Critical: immediate alert
        sendImmediateBLEAlert(&event);
        soundBuzzer(CRITICAL_PATTERN);
        
    } else if (predicted_class == AFIB || predicted_class == PVC) {
        // Non-critical: batch upload
        event_count++;
        if (event_count >= 10) {
            scheduleBatchUpload();
        }
    }
}

void sendImmediateBLEAlert(AnomalyEvent_t* event) {
    // Ensure BLE is active
    ble_wakeup();
    
    // Send priority alert characteristic
    ble_notifyCharacteristic(ALERT_CHAR_UUID, event, sizeof(AnomalyEvent_t));
    
    // Wait for acknowledgment
    uint8_t retries = 3;
    while (retries-- && !ble_isAcknowledged()) {
        delay_ms(1000);
        ble_notifyCharacteristic(ALERT_CHAR_UUID, event, sizeof(AnomalyEvent_t));
    }
}

void scheduleBatchUpload() {
    if (ble_isConnected()) {
        uploadPendingEvents();
    } else {
        // Start advertising to request connection
        ble_startAdvertising(ADV_INTERVAL_FAST);
    }
}

void uploadPendingEvents() {
    for (uint16_t i = 0; i < event_count; i++) {
        AnomalyEvent_t event = readEventFromFlash(i);
        ble_sendEvent(EVENT_CHAR_UUID, &event);
        delay_ms(100);  // Rate limiting
    }
    event_count = 0;  // Clear pending
}
```

### Phone App (React Native Example)

```javascript
import { BleManager } from 'react-native-ble-plx';

class CardiacMonitorApp {
    
    async connectToDevice(deviceId) {
        const device = await bleManager.connectToDevice(deviceId);
        await device.discoverAllServicesAndCharacteristics();
        
        // Subscribe to alert notifications
        device.monitorCharacteristicForService(
            CARDIAC_SERVICE_UUID,
            ALERT_CHAR_UUID,
            (error, characteristic) => {
                if (error) {
                    console.error('Alert subscription error:', error);
                    return;
                }
                
                const event = this.parseAnomalyEvent(characteristic.value);
                this.handleAnomalyEvent(event);
            }
        );
    }
    
    handleAnomalyEvent(event) {
        const { classification, confidence, timestamp, ecg_segment } = event;
        
        // Store locally
        this.db.saveAnomaly(event);
        
        // Show notification based on severity
        if (classification === 'VT' || classification === 'VFib') {
            this.showCriticalAlert(event);
            this.sendEmergencyNotification(event);
        } else {
            this.showNormalNotification(event);
        }
        
        // Sync to cloud
        this.cloudSync.uploadAnomaly(event);
    }
    
    showCriticalAlert(event) {
        // Full-screen alert
        Alert.alert(
            '⚠️ Critical Arrhythmia Detected',
            `${event.classification} detected at ${formatTime(event.timestamp)}`,
            [
                { text: 'View ECG', onPress: () => this.showECG(event) },
                { text: 'Call Emergency', onPress: () => this.callEmergency() },
                { text: 'Dismiss', style: 'cancel' }
            ],
            { cancelable: false }
        );
        
        // Sound + vibration
        this.playAlertSound();
        Vibration.vibrate([0, 500, 200, 500]);
    }
    
    showNormalNotification(event) {
        // Banner notification
        PushNotification.localNotification({
            title: `${event.classification} Detected`,
            message: `Recorded at ${formatTime(event.timestamp)}`,
            playSound: true,
            vibrate: true
        });
    }
    
    async syncData() {
        // Request batch upload from device
        await device.writeCharacteristicWithResponseForService(
            CARDIAC_SERVICE_UUID,
            CONTROL_CHAR_UUID,
            base64Encode('SYNC')
        );
        
        // Wait for events
        // (handled by characteristic subscription)
    }
}
```

---

## Data Privacy & Security

### Encryption & Authentication

**BLE Pairing:**
- Secure Simple Pairing (SSP)
- PIN code or passkey entry
- Bonding for persistent pairing

**Data Encryption:**
- AES-128 encryption on BLE link
- End-to-end encryption for cloud sync
- Encrypted storage on phone

### Regulatory Compliance

**HIPAA (USA):**
- Encrypted data in transit and at rest
- Access controls and audit logs
- Business Associate Agreements (BAA)

**GDPR (Europe):**
- User consent for data collection
- Right to data export
- Right to be forgotten (data deletion)

**FDA (Medical Device):**
- Class II medical device (if diagnostic claims)
- Cybersecurity requirements
- Post-market surveillance

### User Privacy Controls

**Configurable:**
- Enable/disable cloud sync
- Choose what data to share
- Set caregiver access permissions
- Auto-delete old data (retention policy)

---

## Clinical Integration

### Doctor Dashboard

**Web Portal Features:**
- View all patients
- Real-time alert feed
- ECG viewer with annotations
- Trend reports (AFib burden, PVC frequency)
- Export for EMR

**Remote Monitoring:**
- Get alerts when patient has event
- Review ECG remotely
- Adjust medication based on trends
- Schedule follow-up appointments

### EMR Integration

**Standards:**
- HL7 FHIR resources
- Observation resource for ECG
- DiagnosticReport for interpretations

**Example FHIR Resource:**
```json
{
  "resourceType": "Observation",
  "status": "final",
  "code": {
    "coding": [{
      "system": "http://loinc.org",
      "code": "8867-4",
      "display": "Heart rate"
    }]
  },
  "effectiveDateTime": "2026-04-19T14:30:00Z",
  "valueQuantity": {
    "value": 145,
    "unit": "beats/minute"
  },
  "interpretation": [{
    "coding": [{
      "system": "http://snomed.info/sct",
      "code": "426749004",
      "display": "Ventricular tachycardia"
    }]
  }]
}
```

---

## Summary Table

| Specification | Value |
|--------------|-------|
| **Storage per event** | 8-50 KB (configurable) |
| **MCU flash capacity** | 240-480 events (2-4 MB) |
| **External flash capacity** | 15,000+ events (128+ MB) |
| **Transmission protocol** | Bluetooth LE (BLE 5.0+) |
| **BLE data rate** | 1 Mbps (125 KB/s typical) |
| **Critical alert latency** | <1 second |
| **Batch sync frequency** | Every 1-4 hours |
| **BLE power consumption** | ~30 mAh/day |
| **Total battery life** | 9-11 days (500 mAh battery) |
| **Phone compatibility** | iOS 10+ / Android 5.0+ |
| **Cloud sync** | Optional, HIPAA/GDPR compliant |
| **EMR integration** | HL7 FHIR standard |

---

## Conclusion

**Yes, storing and transmitting anomaly data to a phone is highly feasible and practical!**

**Key Capabilities:**
- ✅ Store 240-480 events in MCU flash (or 15,000+ with external storage)
- ✅ Real-time critical alerts (<1s latency via BLE)
- ✅ Batch sync for non-critical events (power efficient)
- ✅ Full ECG waveform + features + classification
- ✅ Minimal battery impact (~30 mAh/day for BLE)
- ✅ iOS/Android app with visualization
- ✅ Cloud sync for doctor review
- ✅ EMR integration (HL7 FHIR)

**Benefits:**
- 🏥 Enables remote patient monitoring
- 📊 Provides clinical validation data
- 🚨 Immediate life-threatening alerts
- 📈 Long-term trend analysis
- 👨‍⚕️ Doctor can review ECG remotely
- 📱 Patient has full access to their data

This is a **core feature** for any serious cardiac monitoring wearable and should be prioritized in the design!

---

**Generated:** April 19, 2026
