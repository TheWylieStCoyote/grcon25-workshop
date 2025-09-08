#!/usr/bin/env python3
"""
Protocol Implementation Example
Advanced example showing complete protocol stack implementation
"""

import numpy as np
from gnuradio import gr
import pmt
import struct
import zlib
from enum import Enum

class ProtocolState(Enum):
    """Protocol state machine states"""
    IDLE = 0
    SYNC_SEARCH = 1
    HEADER_DECODE = 2
    PAYLOAD_RECEIVE = 3
    CRC_CHECK = 4
    ACK_WAIT = 5
    ACK_SEND = 6


class ProtocolFramer(gr.sync_block):
    """
    Advanced protocol framer with multiple frame types
    Implements a complete protocol stack with:
    - Multiple frame types (DATA, ACK, CONTROL, BEACON)
    - Error detection (CRC32)
    - Sequence numbering
    - Fragmentation support
    """
    
    def __init__(self, mtu=256, node_id=1):
        """
        Initialize protocol framer
        
        Args:
            mtu: Maximum transmission unit (bytes)
            node_id: Node identifier for this station
        """
        gr.sync_block.__init__(
            self,
            name='Protocol Framer',
            in_sig=None,
            out_sig=[np.uint8]
        )
        
        self.mtu = mtu
        self.node_id = node_id
        
        # Frame types
        self.FRAME_DATA = 0x01
        self.FRAME_ACK = 0x02
        self.FRAME_CONTROL = 0x03
        self.FRAME_BEACON = 0x04
        
        # Protocol parameters
        self.preamble = np.array([0xAA] * 8, dtype=np.uint8)  # Alternating bits
        self.sync_word = np.array([0xD3, 0x91, 0x52, 0x67], dtype=np.uint8)
        
        # Sequence numbering
        self.tx_sequence = 0
        self.rx_sequence = 0
        
        # Fragmentation
        self.fragment_buffer = {}
        self.max_fragments = 16
        
        # Output queue
        self.output_queue = []
        
        # Statistics
        self.frames_sent = 0
        self.bytes_sent = 0
        
        # Message ports
        self.message_port_register_in(pmt.intern("pdus"))
        self.message_port_register_out(pmt.intern("tx_info"))
        self.set_msg_handler(pmt.intern("pdus"), self.handle_pdu)
    
    def create_frame(self, frame_type, payload, dest_id=0xFF, flags=0):
        """
        Create a complete frame with header and CRC
        
        Frame structure:
        - Preamble (8 bytes)
        - Sync word (4 bytes)
        - Frame type (1 byte)
        - Flags (1 byte)
        - Source ID (2 bytes)
        - Dest ID (2 bytes)
        - Sequence (2 bytes)
        - Length (2 bytes)
        - Payload (variable)
        - CRC32 (4 bytes)
        """
        frame = []
        
        # Add preamble and sync
        frame.extend(self.preamble)
        frame.extend(self.sync_word)
        
        # Frame header
        header = struct.pack('>BBHHHHH',
            frame_type,                    # Frame type
            flags,                          # Flags
            self.node_id,                   # Source ID
            dest_id,                        # Destination ID
            self.tx_sequence,               # Sequence number
            0,                              # Fragment info (packed later)
            len(payload)                    # Payload length
        )
        frame.extend(header)
        
        # Add payload
        frame.extend(payload)
        
        # Calculate and add CRC32
        crc = zlib.crc32(bytes(header + payload))
        frame.extend(struct.pack('>I', crc))
        
        # Increment sequence number
        self.tx_sequence = (self.tx_sequence + 1) % 65536
        
        return np.array(frame, dtype=np.uint8)
    
    def fragment_data(self, data, max_fragment_size):
        """
        Fragment large data into smaller pieces
        """
        fragments = []
        total_fragments = (len(data) + max_fragment_size - 1) // max_fragment_size
        
        for i in range(total_fragments):
            start = i * max_fragment_size
            end = min(start + max_fragment_size, len(data))
            
            fragment_info = (i << 8) | total_fragments  # Fragment number and total
            fragments.append({
                'data': data[start:end],
                'fragment_info': fragment_info
            })
        
        return fragments
    
    def handle_pdu(self, msg):
        """Handle incoming PDU for transmission"""
        try:
            # Extract PDU
            pdu = pmt.to_python(msg)
            
            if isinstance(pdu, tuple) and len(pdu) == 2:
                meta = pdu[0] if pdu[0] else {}
                data = np.array(pdu[1], dtype=np.uint8)
            else:
                data = np.array(pdu, dtype=np.uint8)
                meta = {}
            
            # Get metadata
            frame_type = meta.get('frame_type', self.FRAME_DATA)
            dest_id = meta.get('dest_id', 0xFFFF)  # Broadcast by default
            priority = meta.get('priority', 0)
            
            # Check if fragmentation needed
            if len(data) > self.mtu - 20:  # Account for header overhead
                fragments = self.fragment_data(data, self.mtu - 20)
                for frag in fragments:
                    frame = self.create_frame(
                        frame_type,
                        frag['data'],
                        dest_id,
                        flags=frag['fragment_info']
                    )
                    self.output_queue.append(frame)
            else:
                # Single frame
                frame = self.create_frame(frame_type, data, dest_id)
                self.output_queue.append(frame)
            
            # Update statistics
            self.frames_sent += 1
            self.bytes_sent += len(data)
            
            # Publish transmission info
            tx_info = {
                'frames_sent': self.frames_sent,
                'bytes_sent': self.bytes_sent,
                'queue_size': len(self.output_queue)
            }
            self.message_port_pub(
                pmt.intern("tx_info"),
                pmt.to_pmt(tx_info)
            )
            
        except Exception as e:
            print(f"Error handling PDU: {e}")
    
    def work(self, input_items, output_items):
        """
        Output queued frames
        """
        out = output_items[0]
        
        if not self.output_queue:
            # No data to send, output zeros
            out[:] = 0
            return len(out)
        
        # Output as many complete frames as possible
        output_index = 0
        while self.output_queue and output_index < len(out):
            frame = self.output_queue[0]
            
            if output_index + len(frame) <= len(out):
                # Frame fits, output it
                out[output_index:output_index + len(frame)] = frame
                output_index += len(frame)
                self.output_queue.pop(0)
            else:
                # Frame doesn't fit, save for next call
                break
        
        # Fill remainder with zeros
        if output_index < len(out):
            out[output_index:] = 0
        
        return len(out)


class ProtocolDeframer(gr.sync_block):
    """
    Advanced protocol deframer with state machine
    Handles frame synchronization, validation, and reassembly
    """
    
    def __init__(self, node_id=1):
        """
        Initialize protocol deframer
        
        Args:
            node_id: Node identifier for this station
        """
        gr.sync_block.__init__(
            self,
            name='Protocol Deframer',
            in_sig=[np.uint8],
            out_sig=None
        )
        
        self.node_id = node_id
        
        # Sync pattern
        self.sync_word = np.array([0xD3, 0x91, 0x52, 0x67], dtype=np.uint8)
        self.sync_threshold = 3  # Allow 1 bit error
        
        # State machine
        self.state = ProtocolState.IDLE
        self.sync_buffer = np.zeros(len(self.sync_word), dtype=np.uint8)
        
        # Frame reception
        self.frame_buffer = []
        self.expected_length = 0
        self.header = None
        
        # Fragment reassembly
        self.fragment_buffer = {}
        self.reassembly_timeout = 1000  # samples
        
        # Statistics
        self.frames_received = 0
        self.frames_dropped = 0
        self.crc_errors = 0
        
        # Message ports
        self.message_port_register_out(pmt.intern("pdus"))
        self.message_port_register_out(pmt.intern("rx_info"))
    
    def check_sync(self, pattern):
        """Check if pattern matches sync word within threshold"""
        errors = np.sum(pattern != self.sync_word)
        return errors <= (len(self.sync_word) - self.sync_threshold)
    
    def process_header(self, header_bytes):
        """Process frame header"""
        try:
            header = struct.unpack('>BBHHHHH', bytes(header_bytes))
            return {
                'frame_type': header[0],
                'flags': header[1],
                'source_id': header[2],
                'dest_id': header[3],
                'sequence': header[4],
                'fragment_info': header[5],
                'length': header[6]
            }
        except struct.error:
            return None
    
    def check_crc(self, frame):
        """Verify frame CRC"""
        if len(frame) < 4:
            return False
        
        payload = frame[:-4]
        received_crc = struct.unpack('>I', bytes(frame[-4:]))[0]
        calculated_crc = zlib.crc32(bytes(payload))
        
        return received_crc == calculated_crc
    
    def reassemble_fragments(self, source_id, sequence, fragment_info, data):
        """Reassemble fragmented data"""
        fragment_num = (fragment_info >> 8) & 0xFF
        total_fragments = fragment_info & 0xFF
        
        if total_fragments == 0:
            # Not fragmented
            return data
        
        # Create key for fragment group
        key = (source_id, sequence - fragment_num)
        
        if key not in self.fragment_buffer:
            self.fragment_buffer[key] = {
                'fragments': {},
                'total': total_fragments,
                'timestamp': self.frames_received
            }
        
        # Add fragment
        self.fragment_buffer[key]['fragments'][fragment_num] = data
        
        # Check if all fragments received
        if len(self.fragment_buffer[key]['fragments']) == total_fragments:
            # Reassemble
            complete_data = []
            for i in range(total_fragments):
                if i in self.fragment_buffer[key]['fragments']:
                    complete_data.extend(self.fragment_buffer[key]['fragments'][i])
                else:
                    # Missing fragment
                    return None
            
            # Clean up
            del self.fragment_buffer[key]
            return complete_data
        
        return None  # Still waiting for fragments
    
    def work(self, input_items, output_items):
        """
        Process input bytes and extract frames
        """
        input_bytes = input_items[0]
        
        for byte in input_bytes:
            if self.state == ProtocolState.IDLE:
                # Look for sync pattern
                self.sync_buffer = np.roll(self.sync_buffer, -1)
                self.sync_buffer[-1] = byte
                
                if self.check_sync(self.sync_buffer):
                    self.state = ProtocolState.HEADER_DECODE
                    self.frame_buffer = []
                    self.expected_length = 14  # Header size
            
            elif self.state == ProtocolState.HEADER_DECODE:
                # Collect header bytes
                self.frame_buffer.append(byte)
                
                if len(self.frame_buffer) >= self.expected_length:
                    # Parse header
                    self.header = self.process_header(self.frame_buffer)
                    
                    if self.header:
                        # Check if frame is for us
                        if (self.header['dest_id'] == self.node_id or 
                            self.header['dest_id'] == 0xFFFF):  # Broadcast
                            
                            self.expected_length = self.header['length'] + 4  # Payload + CRC
                            self.state = ProtocolState.PAYLOAD_RECEIVE
                        else:
                            # Not for us, ignore
                            self.state = ProtocolState.IDLE
                    else:
                        # Invalid header
                        self.frames_dropped += 1
                        self.state = ProtocolState.IDLE
            
            elif self.state == ProtocolState.PAYLOAD_RECEIVE:
                # Collect payload and CRC
                self.frame_buffer.append(byte)
                
                if len(self.frame_buffer) >= 14 + self.expected_length:
                    # Complete frame received
                    self.state = ProtocolState.CRC_CHECK
            
            if self.state == ProtocolState.CRC_CHECK:
                # Verify CRC
                frame_data = self.frame_buffer[14:]  # Skip header
                
                if self.check_crc(frame_data):
                    # Valid frame
                    payload = frame_data[:-4]  # Remove CRC
                    
                    # Handle fragmentation
                    complete_data = self.reassemble_fragments(
                        self.header['source_id'],
                        self.header['sequence'],
                        self.header['fragment_info'],
                        payload
                    )
                    
                    if complete_data is not None:
                        # Output PDU
                        meta = pmt.dict_add(
                            pmt.make_dict(),
                            pmt.intern("frame_type"),
                            pmt.from_long(self.header['frame_type'])
                        )
                        meta = pmt.dict_add(
                            meta,
                            pmt.intern("source_id"),
                            pmt.from_long(self.header['source_id'])
                        )
                        
                        self.message_port_pub(
                            pmt.intern("pdus"),
                            pmt.cons(meta, pmt.init_u8vector(len(complete_data), complete_data))
                        )
                        
                        self.frames_received += 1
                else:
                    # CRC error
                    self.crc_errors += 1
                    self.frames_dropped += 1
                
                # Publish statistics
                if self.frames_received % 100 == 0:
                    rx_info = {
                        'frames_received': self.frames_received,
                        'frames_dropped': self.frames_dropped,
                        'crc_errors': self.crc_errors,
                        'fragment_groups': len(self.fragment_buffer)
                    }
                    self.message_port_pub(
                        pmt.intern("rx_info"),
                        pmt.to_pmt(rx_info)
                    )
                
                self.state = ProtocolState.IDLE
        
        # Clean up old fragments
        if len(self.fragment_buffer) > 0:
            current_time = self.frames_received
            keys_to_remove = []
            
            for key, group in self.fragment_buffer.items():
                if current_time - group['timestamp'] > self.reassembly_timeout:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.fragment_buffer[key]
                self.frames_dropped += 1
        
        return len(input_bytes)


class ARQController(gr.sync_block):
    """
    Automatic Repeat Request (ARQ) controller
    Implements Stop-and-Wait, Go-Back-N, and Selective Repeat ARQ
    """
    
    def __init__(self, arq_type='selective', window_size=8, timeout=1000):
        """
        Initialize ARQ controller
        
        Args:
            arq_type: Type of ARQ ('stop_wait', 'go_back_n', 'selective')
            window_size: Window size for sliding window protocols
            timeout: Retransmission timeout in samples
        """
        gr.sync_block.__init__(
            self,
            name='ARQ Controller',
            in_sig=None,
            out_sig=None
        )
        
        self.arq_type = arq_type
        self.window_size = window_size
        self.timeout = timeout
        
        # Transmission window
        self.tx_window = {}
        self.tx_next_seq = 0
        self.tx_base = 0
        
        # Reception window
        self.rx_window = {}
        self.rx_expected = 0
        
        # Timers
        self.timers = {}
        self.current_time = 0
        
        # Statistics
        self.packets_sent = 0
        self.packets_received = 0
        self.retransmissions = 0
        self.acks_sent = 0
        self.acks_received = 0
        
        # Message ports
        self.message_port_register_in(pmt.intern("tx_pdus"))
        self.message_port_register_in(pmt.intern("rx_pdus"))
        self.message_port_register_in(pmt.intern("acks"))
        self.message_port_register_out(pmt.intern("pdus_out"))
        self.message_port_register_out(pmt.intern("acks_out"))
        self.message_port_register_out(pmt.intern("arq_stats"))
        
        self.set_msg_handler(pmt.intern("tx_pdus"), self.handle_tx_pdu)
        self.set_msg_handler(pmt.intern("rx_pdus"), self.handle_rx_pdu)
        self.set_msg_handler(pmt.intern("acks"), self.handle_ack)
    
    def handle_tx_pdu(self, msg):
        """Handle PDU for transmission with ARQ"""
        if self.arq_type == 'stop_wait':
            self.handle_tx_stop_wait(msg)
        elif self.arq_type == 'go_back_n':
            self.handle_tx_go_back_n(msg)
        else:  # selective repeat
            self.handle_tx_selective(msg)
    
    def handle_tx_stop_wait(self, msg):
        """Stop-and-Wait ARQ transmission"""
        # Wait for previous packet to be acknowledged
        if len(self.tx_window) > 0:
            return  # Still waiting for ACK
        
        # Add sequence number and store
        seq = self.tx_next_seq
        self.tx_window[seq] = msg
        self.timers[seq] = self.current_time + self.timeout
        
        # Send packet
        meta = pmt.dict_add(
            pmt.make_dict(),
            pmt.intern("seq"),
            pmt.from_long(seq)
        )
        self.message_port_pub(pmt.intern("pdus_out"), pmt.cons(meta, msg))
        
        self.packets_sent += 1
        self.tx_next_seq = (self.tx_next_seq + 1) % 256
    
    def handle_tx_go_back_n(self, msg):
        """Go-Back-N ARQ transmission"""
        # Check window
        if len(self.tx_window) >= self.window_size:
            return  # Window full
        
        seq = self.tx_next_seq
        self.tx_window[seq] = msg
        self.timers[seq] = self.current_time + self.timeout
        
        # Send packet
        meta = pmt.dict_add(
            pmt.make_dict(),
            pmt.intern("seq"),
            pmt.from_long(seq)
        )
        self.message_port_pub(pmt.intern("pdus_out"), pmt.cons(meta, msg))
        
        self.packets_sent += 1
        self.tx_next_seq = (self.tx_next_seq + 1) % 256
    
    def handle_tx_selective(self, msg):
        """Selective Repeat ARQ transmission"""
        # Similar to Go-Back-N but with selective retransmission
        if len(self.tx_window) >= self.window_size:
            return
        
        seq = self.tx_next_seq
        self.tx_window[seq] = msg
        self.timers[seq] = self.current_time + self.timeout
        
        meta = pmt.dict_add(
            pmt.make_dict(),
            pmt.intern("seq"),
            pmt.from_long(seq)
        )
        self.message_port_pub(pmt.intern("pdus_out"), pmt.cons(meta, msg))
        
        self.packets_sent += 1
        self.tx_next_seq = (self.tx_next_seq + 1) % 256
    
    def handle_rx_pdu(self, msg):
        """Handle received PDU"""
        try:
            meta = pmt.car(msg) if pmt.is_pair(msg) else pmt.make_dict()
            seq = pmt.to_long(pmt.dict_ref(meta, pmt.intern("seq"), pmt.from_long(0)))
            
            if self.arq_type == 'stop_wait':
                # Accept if expected sequence
                if seq == self.rx_expected:
                    self.packets_received += 1
                    self.rx_expected = (self.rx_expected + 1) % 256
                    
                    # Send ACK
                    self.send_ack(seq)
                    
                    # Forward PDU
                    self.message_port_pub(pmt.intern("pdus_out"), msg)
            
            elif self.arq_type == 'go_back_n':
                # Accept in-order packets
                if seq == self.rx_expected:
                    self.packets_received += 1
                    self.rx_expected = (self.rx_expected + 1) % 256
                    self.send_ack(seq)
                    self.message_port_pub(pmt.intern("pdus_out"), msg)
                elif seq < self.rx_expected:
                    # Duplicate, send ACK anyway
                    self.send_ack(seq)
            
            else:  # selective repeat
                # Accept within window
                window_start = self.rx_expected
                window_end = (window_start + self.window_size) % 256
                
                if self.in_window(seq, window_start, window_end):
                    self.rx_window[seq] = msg
                    self.send_ack(seq)
                    
                    # Deliver in-order packets
                    while self.rx_expected in self.rx_window:
                        self.message_port_pub(
                            pmt.intern("pdus_out"),
                            self.rx_window[self.rx_expected]
                        )
                        del self.rx_window[self.rx_expected]
                        self.packets_received += 1
                        self.rx_expected = (self.rx_expected + 1) % 256
                        
        except Exception as e:
            print(f"Error handling received PDU: {e}")
    
    def handle_ack(self, msg):
        """Handle received ACK"""
        try:
            ack_seq = pmt.to_long(msg)
            
            if self.arq_type == 'stop_wait':
                # Clear window if ACK matches
                if ack_seq in self.tx_window:
                    del self.tx_window[ack_seq]
                    del self.timers[ack_seq]
                    self.acks_received += 1
            
            elif self.arq_type == 'go_back_n':
                # Cumulative ACK
                while self.tx_base != (ack_seq + 1) % 256:
                    if self.tx_base in self.tx_window:
                        del self.tx_window[self.tx_base]
                        del self.timers[self.tx_base]
                    self.tx_base = (self.tx_base + 1) % 256
                    self.acks_received += 1
            
            else:  # selective repeat
                # Individual ACK
                if ack_seq in self.tx_window:
                    del self.tx_window[ack_seq]
                    del self.timers[ack_seq]
                    self.acks_received += 1
                    
                    # Advance window base
                    while self.tx_base not in self.tx_window and \
                          self.tx_base != self.tx_next_seq:
                        self.tx_base = (self.tx_base + 1) % 256
                        
        except Exception as e:
            print(f"Error handling ACK: {e}")
    
    def send_ack(self, seq):
        """Send acknowledgment"""
        self.message_port_pub(
            pmt.intern("acks_out"),
            pmt.from_long(seq)
        )
        self.acks_sent += 1
    
    def in_window(self, seq, start, end):
        """Check if sequence number is in window"""
        if start <= end:
            return start <= seq < end
        else:  # Wraparound
            return seq >= start or seq < end
    
    def check_timeouts(self):
        """Check for timeouts and retransmit"""
        self.current_time += 1
        
        for seq in list(self.timers.keys()):
            if self.current_time >= self.timers[seq]:
                # Timeout occurred
                if self.arq_type == 'go_back_n':
                    # Retransmit all from seq onwards
                    for s in range(seq, self.tx_next_seq):
                        if s in self.tx_window:
                            meta = pmt.dict_add(
                                pmt.make_dict(),
                                pmt.intern("seq"),
                                pmt.from_long(s)
                            )
                            self.message_port_pub(
                                pmt.intern("pdus_out"),
                                pmt.cons(meta, self.tx_window[s])
                            )
                            self.timers[s] = self.current_time + self.timeout
                            self.retransmissions += 1
                else:
                    # Retransmit individual packet
                    if seq in self.tx_window:
                        meta = pmt.dict_add(
                            pmt.make_dict(),
                            pmt.intern("seq"),
                            pmt.from_long(seq)
                        )
                        self.message_port_pub(
                            pmt.intern("pdus_out"),
                            pmt.cons(meta, self.tx_window[seq])
                        )
                        self.timers[seq] = self.current_time + self.timeout
                        self.retransmissions += 1
    
    def work(self, input_items, output_items):
        """
        Periodic processing for timeouts and statistics
        """
        # Check timeouts
        self.check_timeouts()
        
        # Publish statistics periodically
        if self.current_time % 1000 == 0:
            stats = {
                'packets_sent': self.packets_sent,
                'packets_received': self.packets_received,
                'retransmissions': self.retransmissions,
                'acks_sent': self.acks_sent,
                'acks_received': self.acks_received,
                'window_size': len(self.tx_window),
                'efficiency': (self.packets_sent - self.retransmissions) / 
                             max(self.packets_sent, 1)
            }
            
            self.message_port_pub(
                pmt.intern("arq_stats"),
                pmt.to_pmt(stats)
            )
        
        return 0  # No stream processing


def test_protocol():
    """Test protocol implementation"""
    print("Testing Protocol Implementation...")
    
    from gnuradio import blocks
    
    class ProtocolTestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "Protocol Test")
            
            # Create test data
            test_data = b"Hello, this is a test message for the protocol stack!"
            
            # Protocol components
            framer = ProtocolFramer(mtu=32, node_id=1)
            deframer = ProtocolDeframer(node_id=2)
            arq = ARQController(arq_type='selective', window_size=4)
            
            # Data source
            src = blocks.message_strobe(
                pmt.cons(
                    pmt.make_dict(),
                    pmt.init_u8vector(len(test_data), list(test_data))
                ),
                1000  # Period in ms
            )
            
            # Sink for frames
            sink = blocks.vector_sink_b()
            
            # Connect
            self.connect(framer, sink)
            
            # Message connections
            self.msg_connect(src, "strobe", framer, "pdus")
            
            # For complete test, would connect through channel to deframer
    
    # Run test
    tb = ProtocolTestFlowgraph()
    tb.start()
    
    import time
    time.sleep(2)
    
    tb.stop()
    tb.wait()
    
    print("\nProtocol test complete!")


if __name__ == '__main__':
    test_protocol()