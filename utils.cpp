const char *to_byte_arr(std::vector<std::pair<std::string, std::string>> pairs) {
    std::string flattened_pairs;
    
    for (auto pair : pairs)
        flattened_pairs += pair.first + pair.second;
    
    return flattened_pairs.c_str();
}
