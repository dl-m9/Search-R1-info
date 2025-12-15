# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simple script to calculate info gain scores using SEPER service.
"""

import os
import sys
import re

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from verl.utils.reward_score.seper_client import SePerClient


def extract_context_from_solution(solution_str: str) -> str:
    """Extract context from <information> tags in solution string."""
    info_pattern = r'<information>(.*?)</information>'
    info_matches = re.findall(info_pattern, solution_str, re.DOTALL)
    
    if info_matches:
        context = '\n'.join([info.strip() for info in info_matches])
        return context
    return ''


def extract_question_from_solution(solution_str: str) -> str:
    """Extract question from solution string."""
    question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', solution_str, re.IGNORECASE)
    if question_match:
        return question_match.group(1).strip()
    return ''


def main():
    # Get service URL from environment or use default
    service_url = os.getenv('SEPER_SERVICE_URL', 'http://0.0.0.0:0310')
    
    print(f"Connecting to SEPER service: {service_url}")
    client = SePerClient(service_url=service_url)
    
    if not client.health_check():
        print(f"❌ Error: SEPER service at {service_url} is not available!")
        sys.exit(1)
    
    print("✅ SEPER service is available\n")
    
    # Example case from log file
    question = "In what city was the band behind the album Love Bites formed?"
    context = """Doc 1(Title: "Love Bites (band)") Love Bites (band) Love Bites is an English girl band that formed in 2004 and disbanded in 2007, but reformed again in 2011. The band, named after a Def Leppard song, formed while the members were still in school. They started out as a three piece group consisting of Danielle Graham, Aimee Haddon and Hannah Haddon with Nicki Wood later joining the band. They did some touring and recording. In 2006 Wood left the band and was replaced by Beka Pritchard. They broke up the next year. The band released their debut single, "You Broke My Heart", on October 2005.
Doc 2(Title: "Lovebites (band)") Lovebites (band) Lovebites (stylized as LOVEBITES) is a Japanese all-female heavy metal band, formed in 2016 by former Destrose members Miho and Haruna. They derived their name from the song "Love Bites (So Do I)" by Halestorm. Lovebites was formed in Tokyo in 2016 by bassist Miho and drummer Haruna. The two met while members of Destrose, another all-female metal band that disbanded in 2015. After recruiting guitarist Midori and support guitarist and keyboardist Miyako (then known as Mi-Ya), the four chose vocalist Asami based on a demo she made. Midori was in the band Gekijo Metalicche, Miyako was a
Doc 3(Title: "Love Bites (album)") their particular brand of pop and their disillusionment with its restrictions. Producer Martin Rushent clarifies the elements of the sound even further, and Shelley's songwriting continues to improve"". In a retrospective review, BBC Music described it as ""an essential purchase for anyone remotely interested in punk's history."" AllMusic wrote: ""More musically accomplished, more obsessively self-questioning, and with equally energetic yet sometimes gloomy performances, ""Love Bites"" finds the Buzzcocks coming into their own."" Love Bites (album) Love Bites is the second studio album by English punk rock band Buzzcocks. It was released on 22 September 1978, through United Artists, on which album is the band behind the album Love Bites formed?
Doc 1(Title: Buzzcocks) Buzzcocks Buzzcocks are an English punk rock band, formed in Bolton, England, in 1976 by singer-songwriter-guitarist Pete Shelley and singer-songwriter Howard Devoto. They are regarded as a seminal influence on the Manchester music scene, the independent record label movement, punk rock, power pop, and pop punk. They achieved commercial success with singles that fused pop craftsmanship with rapid-fire punk energy. These singles were collected on ""Singles Going Steady"", described by critic Ned Raggett as a ""punk masterpiece"". Devoto and Shelley chose the name ""Buzzcocks"" after reading the headline, ""It's the Buzz, Cock!"", in a review of the TV series ""Rock
Doc 2(Title: Buzzcocks) Buzzcocks Buzzcocks are an English punk rock band, formed in Bolton, England, in 1976 by singer-songwriter-guitarist Pete Shelley and singer-songwriter Howard Devoto. They are regarded as a seminal influence on the Manchester music scene, the independent record label movement, punk rock, power pop, and pop punk. They achieved commercial success with singles that fused pop craftsmanship with rapid-fire punk energy. These singles were collected on ""Singles Going Steady"", described by critic Ned Raggett as a ""punk masterpiece"". Devoto and Shelley chose the name ""Buzzcocks"" after reading the headline, ""It's the Buzz, Cock!"", in a review of the TV series ""Rock
Doc 3(Title: Buzzcocks) the Institute, responded to the notice. Trafford had previously been involved in electronic music, while McNeish had played rock. By late 1975, Trafford and McNeish had recruited a drummer and formed, in effect, an embryonic version of Buzzcocks. The band formed, officially, in February 1976; McNeish assumed the stage name Pete Shelley and Trafford named himself Howard Devoto. They performed live for the first time on 1 April 1976 at their college. Garth Davies played bass guitar and Mick Singleton played drums. Singleton also played in local band Black Cat Bone. After reading an ""NME"" review of the Sex Pistols'"""
    answers = ['Bolton']
    
    print(f"Question: {question}")
    print(f"Context length: {len(context)} characters")
    print(f"Answers: {answers}\n")
    
    # Batch compute: create 20 identical cases
    batch_size = 1
    items = [
        {
            "question": question,
            "context": context,
            "answers": answers
        }
    ] * batch_size
    
    print(f"Computing info gain for {batch_size} identical cases in batch...")
    info_gain_scores = client.compute_info_gain_batch(items)
    
    print(f"\nInfo Gain Scores ({len(info_gain_scores)} results):")
    for i, score in enumerate(info_gain_scores, 1):
        print(f"  {i:2d}. {score:.6f}")
    
    if info_gain_scores:
        import statistics
        print(f"\nStatistics:")
        print(f"  Mean:   {statistics.mean(info_gain_scores):.6f}")
        print(f"  Median: {statistics.median(info_gain_scores):.6f}")
        print(f"  Std:    {statistics.stdev(info_gain_scores):.6f}")
        print(f"  Min:    {min(info_gain_scores):.6f}")
        print(f"  Max:    {max(info_gain_scores):.6f}")
    
    # You can modify the question, context, and answers above to calculate for your cases


if __name__ == '__main__':
    main()
