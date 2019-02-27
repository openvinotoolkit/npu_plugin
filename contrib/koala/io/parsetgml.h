/** \file parsetgml.h
 * \brief Auxiliary class for keeping graph properties in input/output process (optional).
 *
 * Functors that allow us conveniently read/write graph with VertInfo=EdgeInfo=ParSet and graphml */

#ifndef KOALA_PARSET_GML_H
#define KOALA_PARSET_GML_H
#include<map>
#include<string>
#include"text.h"
#include"graphml.h"

namespace Koala {

namespace IO {

// ParSetVertRead=ParSetEdgeRead->ParSetRead, ParSetEdgeWrite=ParSetVertWrite->ParSetWrite
/**\brief Function object that reads GraphML to ParSet.*/
struct ParSetRead {
	ParSetRead(GraphML &g): m_gml(g), m_idkey(), m_readid(false)			{};
	ParSetRead(GraphML &g, std::string ik): m_gml(g), m_idkey(ik), m_readid(true)	{};
	ParSet operator()(GraphMLKeysRead *gmlkr) {
		std::map<std::string, GraphMLKeyTypes::Type> keys;
		std::map<std::string, GraphMLKeyTypes::Type>::iterator it;
		std::string name;
		ParSet p;
		gmlkr->getKeys(keys);
		if(m_readid) p.set(m_idkey, gmlkr->getId().c_str());
		for(it = keys.begin(); it != keys.end(); ++it) {
			name = m_gml.getKeyAttrName(it->first.c_str());
			if(name == "") name = it->first;
			switch(it->second) {
				case GraphMLKeyTypes::Bool: p.set(name.c_str(), gmlkr->getBool(it->first.c_str())); break;
				case GraphMLKeyTypes::Int:
				case GraphMLKeyTypes::Long: p.set(name.c_str(), gmlkr->getInt(it->first.c_str())); break;
				case GraphMLKeyTypes::Float:
				case GraphMLKeyTypes::Double: p.set(name.c_str(), gmlkr->getDouble(it->first.c_str())); break;
				case GraphMLKeyTypes::String: p.set(name.c_str(), gmlkr->getString(it->first.c_str())); break;
				default: assert(0);
				};
			};
		return p;
		};
private:
	GraphML &m_gml;
	std::string m_idkey;
	bool m_readid;
};


template<class Graph>
struct ParSetWrite {

    template <class VertOrEdge>
	void operator()(VertOrEdge vert,
			GraphMLKeysWrite *gmlkw) {
		std::vector<std::string> keys;
		vert->info.getKeys(keys);
		for(int i = 0; i < keys.size(); i++) {
			switch(vert->info.getType(keys[i])) {
				case PST_Bool: gmlkw->setBool(keys[i].c_str(), vert->info.template get<bool>(keys[i])); break;
				case PST_Int: gmlkw->setInt(keys[i].c_str(), vert->info.template get<int>(keys[i])); break;
				case PST_Double: gmlkw->setDouble(keys[i].c_str(), vert->info.template get<double>(keys[i])); break;
				case PST_String: gmlkw->setString(keys[i].c_str(), vert->info.template get<std::string>(keys[i])); break;
				default : ;
				};
			};
		}
	};



}; // namespace IO

}; // namespace Koala

#endif
