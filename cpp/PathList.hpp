/*
 * PATHLIST_HPP_
 *
 *  Created on: 15 may 2023
 *      Author: Mattia D'Emidio
 */

#ifndef PATHLIST_HPP_
#define PATHLIST_HPP_

#include <cassert>
#include <deque>
#include <iostream>     // std::cout
#include <algorithm>    // std::lower_bound, std::upper_bound, std::sort
#include <vector>       // std::vector

struct path_data
{
	std::deque<uint32_t> vertices;
	
	path_data(){}
	path_data(uint32_t r){
		vertices.push_back(r);
	}
	path_data(std::deque<uint32_t>::iterator from, std::deque<uint32_t>::iterator to){
		vertices = std::deque<uint32_t>(from,to);
	}
	bool empty() const{
		return vertices.empty();
	}
	// uint32_t getVertex(size_t t){
	// 	return path[t];
	// }
	void appendleft(uint32_t v){
		vertices.push_front(v);
	}
	void append(uint32_t v){
		vertices.push_back(v);
	}
	bool operator<(path_data const & rhs) const{
		return vertices.size() < rhs.vertices.size(); 
	}

	bool operator==(path_data const & rhs) const{
		return weight() == rhs.weight() && vertices == rhs.vertices;
	}
	
	// bool operator!=(path_data const & rhs) const{
	// 	return weight() != rhs.weight() || (vertices != rhs.path);
	// }
	friend std::ostream& operator<<(std::ostream& os, path_data const & rhs){
		for(size_t t=0;t<rhs.vertices.size();t++)
			if(t!=rhs.vertices.size()-1){
				os << rhs.vertices[t]<< " ";
			}
			else{
				os << rhs.vertices[t];
			}
		return os;
	}
	uint32_t weight() const{
		assert(vertices.size()>0);
		return vertices.size()-1;
	}

};

struct compare_path_heap{
    bool operator()(const path_data& n1, const path_data& n2) const
    {
        return n1.weight() > n2.weight();
    }
};
struct compare_path_heap_hub{
    bool operator()(const std::pair<path_data,uint32_t>& n1, const std::pair<path_data,uint32_t>& n2) const
    {
        return n1.first.weight() > n2.first.weight() || (n1.first.weight() == n2.first.weight() && n1.second>n2.second);
    }
};
struct PathList {
	
	std::vector<path_data> paths;
	PathList(){
		paths.clear();
	};
	PathList(path_data p){
		paths.clear();
		paths.push_back(p);
		assert(this->is_sorted());
		assert(this->is_path_in(p));

	};

	void append_path(path_data p){
		paths.push_back(p);
		assert(this->is_sorted());
		assert(this->is_path_in(p));

	};
	// std::vector<path_data>& getPaths(){
	// 	return paths;
	// }
	// size_t getSize(){
	// 	return paths.size();
	// }
	void pop(){
		paths.pop_back();
	}
	// path_data & getPath(size_t pos) {
	// 	return paths[pos];
	// }
	
	void insert_path(path_data p){
		// Search first element that is longer than p
		if(paths.empty()){
			paths.push_back(p);
			assert(this->is_sorted());
			return;
		}

		
		std::vector<path_data>::iterator upper = std::upper_bound(paths.begin(), paths.end(), p);
		
		paths.insert(upper,p);
		assert(this->is_sorted());
		assert(this->is_path_in(p));
		
	};
	bool is_path_in(path_data p){
		if(paths.empty())
			return false;
		if(std::find(paths.begin(), paths.end(), p)==paths.end())
			return false;
		else 
			return true;

	};
	bool is_sorted() const{
		for (size_t i = 0; i < this->paths.size()-1; i++){
			if(paths[i+1]<paths[i])
				return false;
		}
		return true;
		

	};
	void clean(){
		paths.clear();
	}
	virtual ~PathList(){
		paths.clear();

	};
};
#endif /* PATHLIST_HPP_ */